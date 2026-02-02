import requests
import os
import time
import re

class AudioModerationService:
    def __init__(self):
        self.api_key = os.getenv('ASSEMBLYAI_API_KEY')
        self.base_url = 'https://api.assemblyai.com/v2'
        self.headers = {
            'authorization': self.api_key,
            'content-type': 'application/json'
        }
        
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
        
        # Daftar kata kasar Bahasa Indonesia
        self.indonesian_profanity = {
            # Level 5/5 - Paling kasar
            'jancuk', 'jancok', 'cuk', 'ngentot', 'entot', 'kontol', 'memek', 'cukimai', 'pukimak',
            
            # Level 4/5 - Sangat kasar
            'bangsat', 'bajingan', 'asu', 'coli', 'colmek', 'kampret', 'kimak',
            
            # Level 3/5 - Kasar sedang
            'anjing', 'babi', 'monyet', 'goblok', 'tolol', 'bodoh', 'bego', 'setan', 'iblis',
            'tai', 'taik', 'tae', 'peler', 'jembut', 'pantek', 'pepek', 'perek',
            
            # Level 2/5 - Kasar ringan / slang
            'anjir', 'anjrit', 'anjay', 'bangke', 'geblek', 'udik', 'kampungan', 'brengsek',
            'sialan', 'sial', 'kunyuk', 'cebong', 'kadrun',
            
            # Variasi leetspeak & typo umum
            '4nj1ng', 'b4b1', 'm0ny3t', 'g0bl0k', 't0l0l', 'b3g0', 'b0d0h',
            'anjg', 'bjir', 'njir', 'kontl', 'mmk', 'jmbt', 'anj1ng', 'b4ngs4t'
        }

        # --- FIX: Common speech-to-text mispronunciation / transcription variants ---
        # Key: kata yang mungkin di-transcribe AssemblyAI
        # Value: kata kasar yang sebenarnya
        self.transcription_variants = {
            # memek variants (sering di-transcribe salah)
            'mamek': 'memek',
            'momek': 'memek',
            'memak': 'memek',
            'mmek': 'memek',
            'memex': 'memek',
            'mamak': 'memek',

            # kontol variants
            'kuntol': 'kontol',
            'kontl': 'kontol',
            'kntol': 'kontol',
            'kantol': 'kontol',
            'kuntul': 'kontol',

            # pepek variants
            'papek': 'pepek',
            'pipek': 'pepek',
            'pepeq': 'pepek',
            'ppk': 'pepek',

            # anjing variants
            'anjeng': 'anjing',
            'anjong': 'anjing',
            'anjung': 'anjing',
            'anjg': 'anjing',

            # bangsat variants
            'bangset': 'bangsat',
            'bangsuit': 'bangsat',
            'bangzat': 'bangsat',

            # bajingan variants
            'bajangan': 'bajingan',

            # goblok variants
            'goblek': 'goblok',
            'goblak': 'goblok',
            'gublok': 'goblok',

            # tolol variants
            'tulul': 'tolol',
            'talal': 'tolol',

            # brengsek variants
            'brangsek': 'brengsek',
            'brengsak': 'brengsek',

            # entot variants
            'antot': 'entot',
            'intot': 'entot',
            'untot': 'entot',

            # ngentot variants
            'ngantot': 'ngentot',
            'ngintot': 'ngentot',
            'nguntot': 'ngentot',

            # colmek variants
            'calmek': 'colmek',
            'culmek': 'colmek',

            # kampret variants
            'kompret': 'kampret',
            'kumpret': 'kampret',

            # kimak variants
            'kamak': 'kimak',
            'komak': 'kimak',

            # pukimak variants
            'pokimak': 'pukimak',
            'pakimak': 'pukimak',

            # cukimai variants
            'cakimai': 'cukimai',
            'cokimai': 'cukimai',
        }
    
    def upload_audio(self, file_bytes):
        """
        Upload file audio ke AssemblyAI dan return upload_url.
        """
        print("[DEBUG] Uploading audio to AssemblyAI...")
        
        upload_headers = {
            'authorization': self.api_key,
            'content-type': 'application/octet-stream'
        }
        
        response = requests.post(
            f'{self.base_url}/upload',
            headers=upload_headers,
            data=file_bytes
        )
        response.raise_for_status()
        
        upload_url = response.json().get('upload_url')
        print(f"[DEBUG] Upload successful. URL: {upload_url}")
        
        return upload_url

    def _check_indonesian_profanity(self, text):
        """
        Cek kata kasar Bahasa Indonesia dalam teks.
        Sekarang juga handle transcription variants (misspelling dari speech-to-text).
        
        Returns:
            tuple: (ada_kata_kasar, list_kata_yang_terdeteksi)
        """
        if not text:
            return False, []
        
        # Lowercase & hapus karakter spesial (kecuali spasi)
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        detected = []
        for word in words:
            # --- Cek 1: Exact match ke profanity list ---
            if word in self.indonesian_profanity:
                detected.append(word)
                continue
            
            # --- Cek 2: Exact match ke transcription variants ---
            # Ini handle kasus "mamek" -> "memek", dst.
            if word in self.transcription_variants:
                detected.append(f"{word} (transcribed as: {self.transcription_variants[word]})")
                continue

            # --- Cek 3: Substring match ke profanity list ---
            found_substring = False
            for profane in self.indonesian_profanity:
                if profane in word and len(profane) > 3:
                    detected.append(f"{word} (contains: {profane})")
                    found_substring = True
                    break
            if found_substring:
                continue

            # --- Cek 4: Substring match ke transcription variants ---
            # Handle kasus kayak "mameknya" yang mengandung "mamek"
            for variant, original in self.transcription_variants.items():
                if variant in word and len(variant) > 3:
                    detected.append(f"{word} (contains variant: {variant} -> {original})")
                    break
        
        return len(detected) > 0, list(set(detected))
    
    def moderate(self, audio_url, language='id'):
        """
        Moderate audio content using AssemblyAI.
        """
        try:
            print(f"[DEBUG] Starting moderation for URL: {audio_url}")
            print(f"[DEBUG] Language: {language}")
            print(f"[DEBUG] API Key present: {bool(self.api_key)}")
            
            # Submit transcription
            request_data = {
                'audio_url': audio_url,
                'language_code': language
            }
            
            if language == 'en':
                request_data['filter_profanity'] = True
            
            print(f"[DEBUG] Request data: {request_data}")
            
            transcript_response = requests.post(
                f'{self.base_url}/transcript',
                headers=self.headers,
                json=request_data
            )
            
            print(f"[DEBUG] Response status: {transcript_response.status_code}")
            transcript_response.raise_for_status()
            transcript_id = transcript_response.json()['id']
            print(f"[DEBUG] Transcript ID: {transcript_id}")
            
            # Poll for completion
            max_wait = 300
            elapsed = 0
            while elapsed < max_wait:
                polling_response = requests.get(
                    f'{self.base_url}/transcript/{transcript_id}',
                    headers=self.headers
                )
                polling_response.raise_for_status()
                result = polling_response.json()
                
                if result['status'] == 'completed':
                    break
                elif result['status'] == 'error':
                    return {
                        'status': 'error',
                        'message': result.get('error', 'Transcription failed')
                    }
                
                time.sleep(3)
                elapsed += 3
            
            if elapsed >= max_wait:
                return {
                    'status': 'error',
                    'message': 'Transcription timeout (max 5 minutes)'
                }
            
            # Ambil hasil transcription
            text = result.get('text', '')
            words = result.get('words', [])
            
            print(f"[DEBUG] Transcript: '{text}'")
            
            profanity_detected = False
            detections = []
            
            if language == 'en':
                for word_info in words:
                    word = word_info.get('text', '')
                    if '*' in word:
                        profanity_detected = True
                        detections.append({
                            'word': word,
                            'start': word_info.get('start'),
                            'end': word_info.get('end'),
                            'confidence': word_info.get('confidence'),
                            'type': 'english_profanity'
                        })
            else:
                # Indonesian: cek manual pake profanity list + transcription variants
                has_profanity, profane_words = self._check_indonesian_profanity(text)
                print(f"[DEBUG] Profanity check result: {has_profanity}, words: {profane_words}")
                
                if has_profanity:
                    profanity_detected = True
                    for word_info in words:
                        word_lower = word_info.get('text', '').lower()
                        word_clean = re.sub(r'[^\w]', '', word_lower)
                        
                        for profane in profane_words:
                            # Ambil kata asli sebelum " (" annotation
                            profane_clean = profane.split(' (')[0]
                            
                            if word_clean == profane_clean or profane_clean in word_clean:
                                detections.append({
                                    'word': word_info.get('text', ''),
                                    'matched': profane,
                                    'start': word_info.get('start'),
                                    'end': word_info.get('end'),
                                    'confidence': word_info.get('confidence'),
                                    'type': 'indonesian_profanity'
                                })
                                break
            
            # Decision
            if profanity_detected:
                decision = 'auto_reject'
            else:
                decision = 'auto_approve'

            moderation_result = {
                'status': 'success',
                'flagged': profanity_detected,
                'decision': decision,
                'language': language,
                'categories': {
                    'profanity': profanity_detected
                },
                'detections': detections,
                'transcript': text,
                'total_words': len(words),
                'profane_count': len(detections)
            }
            
            return moderation_result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }