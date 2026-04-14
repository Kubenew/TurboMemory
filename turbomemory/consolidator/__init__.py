#!/usr/bin/env python3
import os
import json
import re
import numpy as np
from datetime import datetime, timezone


VAGUE_PATTERNS = [
    (r'\b(maybe|perhaps|possibly|might|could|seems like)\b', ''),
    (r'\b(I think|I believe|it seems|apparently)\b', ''),
    (r'\b(some|a few|several|many|most)\b(\\s+\\w+)', 'multiple\\2'),
    (r'\b(a lot of|lots of)\b(\\s+\\w+)', 'many\\2'),
    (r'\b(kind of|sort of|type of)\b', ''),
    (r'\bmore or less\b', ''),
    (r'\broughly|approximately|about\b', ''),
]

ABSOLUTE_REPLACEMENTS = {
    'maybe': '',
    'perhaps': '',
    'possibly': '',
    'might': '',
    'could': '',
    'seems like': '',
    'I think': '',
    'I believe': '',
    'it seems': '',
    'apparently': '',
    'kind of': '',
    'sort of': '',
    'type of': '',
    'more or less': '',
}


def make_absolute(text: str) -> str:
    result = text

    for pattern, replacement in VAGUE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    for vague, replacement in ABSOLUTE_REPLACEMENTS.items():
        result = result.replace(vague, replacement)

    result = re.sub(r'\n+', '\n', result).strip()

    if result:
        result = result[0].upper() + result[1:]

    return result


def merge_similar_chunks(chunks: list, similarity_threshold: float = 0.85):
    from turbomemory import dequantize_packed, cosine_similarity

    if len(chunks) < 2:
        return chunks

    merged = []
    used = set()

    for i, c1 in enumerate(chunks):
        if i in used:
            continue

        similar = [c1]
        used.add(i)

        for j, c2 in enumerate(chunks):
            if j in used:
                continue
            if j <= i:
                continue

            emb1 = dequantize_packed(c1['embedding_q'])
            emb2 = dequantize_packed(c2['embedding_q'])
            sim = cosine_similarity(emb1, emb2)

            if sim >= similarity_threshold:
                similar.append(c2)
                used.add(j)

        if len(similar) > 1:
            similar.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
            merged_chunk = similar[0].copy()

            total_conf = sum(c.get('confidence', 0.5) for c in similar)
            merged_chunk['confidence'] = total_conf / len(similar)

            all_refs = []
            for c in similar:
                all_refs.extend(c.get('source_refs', []))
            merged_chunk['source_refs'] = list(set(all_refs))

            merged_chunk['text'] = make_absolute(merged_chunk['text'])

            merged_chunk['timestamp'] = min(c.get('timestamp', '') for c in similar)

            merged.append(merged_chunk)
        else:
            merged.append(c1)

    return merged


def resolve_contradictions(chunks: list, decay_factor: float = 0.5) -> list:
    if len(chunks) < 2:
        return chunks

    resolved = []
    for i, c1 in enumerate(chunks):
        is_contradicted = False
        for j, c2 in enumerate(chunks):
            if i == j:
                continue
            if _texts_contradict(c1.get('text', ''), c2.get('text', '')):
                if c1.get('timestamp', '') < c2.get('timestamp', ''):
                    is_contradicted = True
                    break

        if is_contradicted:
            c1['confidence'] *= decay_factor
            c1['staleness'] = min(1.0, c1.get('staleness', 0.0) + 0.3)

        resolved.append(c1)

    return resolved


def _texts_contradict(text1: str, text2: str) -> bool:
    t1 = text1.lower()
    t2 = text2.lower()

    neg_words = ['not', 'never', 'no', 'without', 'can', 'cannot', 'won', 'doesn']
    t1_neg = any(w in t1 for w in neg_words)
    t2_neg = any(w in t2 for w in neg_words)

    if t1_neg != t2_neg:
        w1 = set(w for w in t1.split() if len(w) > 3)
        w2 = set(w for w in t2.split() if len(w) > 3)
        if len(w1.intersection(w2)) >= 3:
            return True

    return False


def consolidate_topic(
    tm,
    topic: str,
    similarity_threshold: float = 0.93,
    min_entropy: float = 0.10,
    staleness_prune: float = 0.90,
    max_chunks: int = 300,
    merge_threshold: float = 0.85,
    convert_to_absolute: bool = True,
) -> dict:
    from turbomemory import quantize_packed, dequantize_packed

    topic_data = tm.load_topic(topic)
    chunks = topic_data.get('chunks', [])

    if not chunks:
        return {'topic': topic, 'before': 0, 'after': 0, 'removed': 0, 'merged': 0, 'absolute': 0}

    before_count = len(chunks)
    merged_count = 0
    absolute_count = 0

    chunks = resolve_contradictions(chunks)

    pruned = []
    for c in chunks:
        if c.get('entropy', 0.0) < min_entropy:
            continue
        if c.get('staleness', 0.0) > staleness_prune:
            continue
        pruned.append(c)

    if merge_threshold < 1.0:
        merged_chunks = merge_similar_chunks(pruned, similarity_threshold=merge_threshold)
        merged_count = len(pruned) - len(merged_chunks)
        pruned = merged_chunks

    if convert_to_absolute:
        for c in pruned:
            original = c.get('text', '')
            c['text'] = make_absolute(c['text'])
            if c['text'] != original:
                absolute_count += 1

    seen_hashes = set()
    deduped = []
    for c in pruned:
        h = tm.sha1_text(c.get('text', ''))
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(c)

    deduped.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
    deduped = deduped[:max_chunks]

    removed = before_count - len(deduped)

    if deduped:
        all_embs = [dequantize_packed(c['embedding_q']) for c in deduped]
        centroid = np.mean(np.vstack(all_embs), axis=0)
        topic_data['centroid_q'] = quantize_packed(centroid, bits=8)
    else:
        topic_data['centroid_q'] = None

    topic_data['chunks'] = deduped
    topic_data['updated'] = datetime.now(timezone.utc).isoformat()

    tm.save_topic(topic_data)

    file_ref = os.path.relpath(tm._topic_path(topic), tm.root)
    tm._upsert_topic_to_db(topic_data, file_ref)
    for c in deduped:
        tm._upsert_chunk_to_db(topic, c)

    tm.log_consolidation(topic, 'consolidate', f'before={before_count}, after={len(deduped)}', removed)

    return {
        'topic': topic,
        'before': before_count,
        'after': len(deduped),
        'removed': removed,
        'merged': merged_count,
        'absolute': absolute_count,
    }


def rewrite_memory_md(tm) -> None:
    header = ['# TurboMemory Index (v0.5)\n']
    lines = header[:]

    for fn in os.listdir(tm.topics_dir):
        if not fn.endswith('.tmem'):
            continue

        path = os.path.join(tm.topics_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                topic_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f'Skipping corrupted topic file {fn}: {e}')
            continue

        topic = topic_data['topic']
        chunks = topic_data.get('chunks', [])

        if not chunks:
            continue

        chunks.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        best = chunks[0]

        entry_id = fn.replace('.tmem', '')[:5]
        score = float(best.get('confidence', 0.5))
        summary = best.get('text', '')[:150]
        file_ref = os.path.relpath(path, tm.root)

        line = f'{entry_id} | {topic} | {summary} | {score:.2f} | {datetime.now(timezone.utc).date()} | {file_ref}\n'
        lines.append(line)

    tm.write_index_lines(lines)


def run_once(tm, args) -> dict:
    topics = []
    for fn in os.listdir(tm.topics_dir):
        if fn.endswith('.tmem'):
            path = os.path.join(tm.topics_dir, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    topics.append(json.load(f)['topic'])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f'Skipping corrupted topic file {fn}: {e}')
                continue

    if not topics:
        logger.info('No topics found.')
        return {'topics': 0, 'total_removed': 0, 'total_merged': 0, 'total_absolute': 0}

    total_removed = 0
    total_merged = 0
    total_absolute = 0

    for t in topics:
        try:
            res = consolidate_topic(
                tm, t,
                similarity_threshold=args.threshold,
                min_entropy=args.min_entropy,
                staleness_prune=args.staleness_prune,
                max_chunks=args.max_chunks,
                merge_threshold=args.merge_threshold,
                convert_to_absolute=args.convert_to_absolute,
            )
            total_removed += res['removed']
            total_merged += res['merged']
            total_absolute += res['absolute']
            logger.info(f'[{t}] {res[\"before\"]} -> {res[\"after\"]} (removed {res[\"removed\"]}, merged {res[\"merged\"]}, absolute {res[\"absolute\"]})')
        except Exception as e:
            logger.error(f'Error consolidating topic {t}: {e}')

    rewrite_memory_md(tm)
    tm.log_consolidation('_global', 'run_complete', f'topics={len(topics)}', total_removed)

    logger.info(f'Done. Removed: {total_removed}, Merged: {total_merged}, Absolute: {total_absolute}')
    return {
        'topics': len(topics),
        'total_removed': total_removed,
        'total_merged': total_merged,
        'total_absolute': total_absolute,
    }


def daemon_loop(tm, args) -> None:
    import time
    logger.info('TurboMemory consolidator daemon started.')
    logger.info(f'interval_sec={args.interval_sec}')

    while True:
        try:
            run_once(tm, args)
        except Exception as e:
            logger.error(f'Consolidation error: {e}')

        time.sleep(args.interval_sec)