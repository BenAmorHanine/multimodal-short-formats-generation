"""
Universal text processing utilities.
Used across Whisper transcription, alignment, analysis.
"""
import whisper


def extract_transcript_with_timestamps(video_path, whisper_model):
    """
    Extract transcript with Whisper.
    
    Used by:
    - ImageBind text modality
    - VideoLLaMA2 text enhancement
    - Subtitle generation
    - Keyword extraction
    """
    result = whisper_model.transcribe(
        video_path,
        word_timestamps=True,
        temperature=0.0,
        verbose=False
    )
    
    segments = []
    for segment in result['segments']:
        segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip()
        })
    
    return segments, result['text']


def align_text_to_segments(transcript_segments, video_segments):
    """
    Align transcript to video segments.
    
    Used by:
    - ImageBind trimodal embedding
    - Any segment-level text processing
    """
    aligned_texts = []
    
    for vid_path, vid_start, vid_end in video_segments:
        overlapping_text = []
        
        for trans_seg in transcript_segments:
            if not (trans_seg['end'] < vid_start or trans_seg['start'] > vid_end):
                overlapping_text.append(trans_seg['text'])
        
        segment_text = " ".join(overlapping_text).strip()
        aligned_texts.append(segment_text if segment_text else "")
    
    return aligned_texts


def extract_keywords(text, top_n=10):
    """
    Extract important keywords from text.
    
    Used by:
    - Highlight detection
    - Content summarization
    """
    # Simple frequency-based (can upgrade to TF-IDF)
    words = text.lower().split()
    from collections import Counter
    
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    filtered = [w for w in words if w not in stopwords and len(w) > 3]
    
    return [word for word, count in Counter(filtered).most_common(top_n)]