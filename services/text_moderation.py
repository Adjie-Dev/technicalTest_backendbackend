import os
import logging
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from cerebras.cloud.sdk import Cerebras


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModerationDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    REVIEW_REQUIRED = "review_required"


class ModerationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ModerationResult:
    status: ModerationStatus
    decision: ModerationDecision
    flagged: bool
    categories: Dict = field(default_factory=dict)
    scores: Dict = field(default_factory=dict)
    matches: list = field(default_factory=list)
    raw_response: Dict = field(default_factory=dict)
    original_lang: str = None
    message: str = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['status'] = self.status.value
        result['decision'] = self.decision.value
        return result
    
    def get(self, key: str, default=None):
        result_dict = self.to_dict()
        return result_dict.get(key, default)


class ManualRulesEngine:
    """Manual keyword-based detection - ALWAYS catches these words"""
    
    # Dictionary of keywords by category
    KEYWORDS = {
        'profanity': [
            'kontol', 'memek', 'pepek', 'puki', 'bangsat', 'bajingan', 
            'brengsek', 'jancuk', 'goblok', 'tolol', 'bego', 'anjing', 
            'asu', 'kampret', 'tai', 'cuk', 'jing', 'anjir', 'bgst',
            'mmk', 'ktl', 'njir', 'njng', 'asw', 'ajg', 'jembut',
            'kimak', 'kntl', 'mmq', 'peler', 'itil', 'tempik', 'tempek',
            'kontl', 'memex', 'pepeq', 'ppk', 'mmek'
        ],
        'sexual': [
            'ngentot', 'ngewe', 'colmek', 'coli', 'lonte', 'pelacur', 
            'jablay', 'sundal', 'sange', 'ngesex', 'entot', 'ewe',
            'memex', 'pepeq', 'ngocok', 'crot', 'crotz', 'crott',
            'jilmek', 'jilat', 'hisap', 'kulum', 'sepong', 'ml',
            'bokep', 'porn', 'pornografi', 'telanjang', 'bugil',
            'entod', 'entut', 'ngentod', 'ngentut', 'crot', 'crooot'
        ],
        'violence': [
            'bunuh', 'tembak', 'hajar', 'pukul', 'tikam', 'bacok',
            'bakar', 'tewas', 'mati', 'sikat', 'gebuk',
            'gampar', 'bogem', 'tendang', 'injak', 'libas', 'tumbuk'
        ],
        'hate_speech': [
            # Existing terms
            'kafir', 'murtad', 'negro', 'nigger', 'keling',
            
            # EXPANDED: Indonesian ethnic slurs and derogatory terms
            # Ethnic targeting with derogatory language
            'jawa hama', 'jawa kotor', 'jawa miskin', 'jawa bodoh',
            'sunda primitif', 'sunda kampungan', 'batak kasar', 'batak biadab',
            'minang pelit', 'padang pelit', 'madura kasar', 'madura brutal',
            'dayak liar', 'papua primitif', 'timor bodoh', 'ambon rusuh',
            'china kafir', 'cina kafir', 'cina pelit', 'china pelit',
            'arab teroris', 'india keling', 'india bau',
            
            # Religious hate speech
            'kristen kafir', 'kristian kafir', 'katolik kafir', 
            'kristen sesat', 'islam teroris', 'muslim teroris',
            'buddha sesat', 'hindu sesat', 'konghucu sesat',
            
            # Dehumanizing terms
            'primitif', 'biadab', 'tidak beradab', 'kampungan',
            'terbelakang', 'barbar',
            
            # Phrases combining ethnicity + negativity
            'hama', 'wabah', 'virus', 'penyakit masyarakat',
            
            # Additional slurs
            'aseng', 'cong', 'encek', 'acong'  # Chinese slurs
        ],
        'offensive': [
            'babi', 'monyet', 'setan', 'iblis', 'bangke', 'busuk',
            'sampah', 'jijik', 'jelek', 'buruk', 'hina',
            'rendah', 'bodoh', 'dungu', 'idiot', 'stupid', 'gila'
        ]
    }
    
    # Contextual hate speech patterns - require combination of words
    HATE_PATTERNS = [
        # Pattern: [ethnicity words] + [negative words]
        {
            'ethnic_terms': ['jawa', 'sunda', 'batak', 'minang', 'padang', 'madura', 
                           'dayak', 'papua', 'timor', 'ambon', 'china', 'cina', 
                           'arab', 'india', 'melayu', 'aceh', 'bugis', 'makassar'],
            'negative_terms': ['hama', 'kotor', 'miskin', 'bodoh', 'primitif', 
                             'kampungan', 'kasar', 'biadab', 'pelit', 'brutal',
                             'liar', 'rusuh', 'teroris', 'bau', 'busuk', 'sampah',
                             'terbelakang', 'barbar', 'tidak beradab', 'virus', 'wabah']
        },
        # Pattern: [religion] + [negative words]
        {
            'ethnic_terms': ['kristen', 'kristian', 'katolik', 'islam', 'muslim', 
                           'buddha', 'hindu', 'konghucu'],
            'negative_terms': ['kafir', 'sesat', 'teroris', 'murtad', 'biadab']
        }
    ]
    
    # Variation patterns (leetspeak)
    CHAR_VARIATIONS = {
        'a': '4a@áàâ',
        'b': 'b8',
        'e': '3eéèê',
        'g': '9g6',
        'i': '1ilíìî!',
        'o': '0oóòô',
        's': '5s$ś',
        't': '7t+',
        'z': '2z',
    }
    
    def __init__(self):
        # Compile regex patterns for each keyword
        self.patterns = {}
        for category, keywords in self.KEYWORDS.items():
            self.patterns[category] = []
            for keyword in keywords:
                # Create pattern that handles variations and spacing
                pattern = self._create_pattern(keyword)
                self.patterns[category].append((keyword, pattern))
    
    def _create_pattern(self, keyword: str) -> re.Pattern:
        """Create regex pattern with variations"""
        pattern_parts = []
        
        for char in keyword.lower():
            if char in self.CHAR_VARIATIONS:
                # Create character class for this letter + variations
                chars = self.CHAR_VARIATIONS[char]
                # Escape special regex chars
                escaped_chars = re.escape(chars)
                pattern_parts.append(f'[{escaped_chars}]')
            else:
                # Regular character, escape it
                pattern_parts.append(re.escape(char))
            
            # Allow optional spaces/dots/dashes between letters
            pattern_parts.append(r'[\s\.\-_\*]*')
        
        # Join parts and remove trailing optional chars
        pattern_str = ''.join(pattern_parts).rstrip(r'[\s\.\-_\*]*')
        
        # Create final pattern with word boundaries
        # Use lookahead/lookbehind for word boundaries to handle edge cases
        final_pattern = r'(?<!\w)' + pattern_str + r'(?!\w)'
        
        try:
            return re.compile(final_pattern, re.IGNORECASE)
        except re.error as e:
            logger.error(f"Regex error for keyword '{keyword}': {e}")
            # Fallback to simple exact match
            return re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
    
    def _check_hate_patterns(self, text: str) -> List[Dict]:
        """Check for contextual hate speech patterns (ethnicity + negative term)"""
        matches = []
        text_lower = text.lower()
        
        for pattern in self.HATE_PATTERNS:
            ethnic_terms = pattern['ethnic_terms']
            negative_terms = pattern['negative_terms']
            
            # Check if text contains both an ethnic term and a negative term
            found_ethnic = [term for term in ethnic_terms if term in text_lower]
            found_negative = [term for term in negative_terms if term in text_lower]
            
            if found_ethnic and found_negative:
                # Check proximity (within 5 words of each other)
                words = text_lower.split()
                for ethnic in found_ethnic:
                    for negative in found_negative:
                        # Find positions
                        ethnic_indices = [i for i, w in enumerate(words) if ethnic in w]
                        negative_indices = [i for i, w in enumerate(words) if negative in w]
                        
                        for e_idx in ethnic_indices:
                            for n_idx in negative_indices:
                                if abs(e_idx - n_idx) <= 5:  # Within 5 words
                                    matches.append({
                                        'type': 'hate_speech',
                                        'match': f"{ethnic} + {negative}",
                                        'original': f"{ethnic} {negative}"
                                    })
                                    break
        
        return matches
    
    def check_text(self, text: str) -> Tuple[Dict[str, bool], Dict[str, float], List[Dict]]:
        """
        Check text against manual rules
        Returns: (categories, scores, matches)
        """
        categories = {cat: False for cat in self.KEYWORDS.keys()}
        scores = {cat: 0.0 for cat in self.KEYWORDS.keys()}
        matches = []
        
        # Normalize text for checking
        normalized_text = text.lower()
        # Also create a version without spaces/special chars for detecting obfuscation
        compact_text = re.sub(r'[\s\.\-_\*\+\^\~]', '', normalized_text)
        
        # Check each category
        for category, patterns in self.patterns.items():
            category_matches = []
            
            for original_keyword, pattern in patterns:
                # Method 1: Regex pattern matching (handles spacing and variations)
                found_regex = pattern.findall(normalized_text)
                if found_regex:
                    categories[category] = True
                    for match in found_regex:
                        category_matches.append({
                            'type': category,
                            'match': match.strip(),
                            'original': original_keyword
                        })
                
                # Method 2: Simple substring check on compact text (backup for regex misses)
                # This catches things like m3m3k, k0nt0l that might slip through
                compact_keyword = original_keyword.replace(' ', '').lower()
                
                # Generate all possible leetspeak variants
                def generate_leetspeak_variants(word):
                    variants = [word]
                    # Common substitutions
                    substitutions = [
                        ('e', '3'), ('o', '0'), ('i', '1'), ('a', '4'),
                        ('s', '5'), ('t', '7'), ('g', '9'), ('b', '8')
                    ]
                    for old, new in substitutions:
                        new_variants = []
                        for v in variants:
                            if old in v:
                                new_variants.append(v.replace(old, new))
                        variants.extend(new_variants)
                    return list(set(variants))  # Remove duplicates
                
                leetspeak_variants = generate_leetspeak_variants(compact_keyword)
                
                for variant in leetspeak_variants:
                    # Check if variant is in compact text
                    if variant in compact_text and variant not in [m['match'] for m in category_matches]:
                        categories[category] = True
                        category_matches.append({
                            'type': category,
                            'match': variant,
                            'original': original_keyword
                        })
                        break  # Found one match, no need to check other variants            
            # Calculate score based on matches
            if category_matches:
                scores[category] = min(1.0, len(category_matches) * 0.3 + 0.7)
                # Deduplicate matches
                unique_matches = []
                seen = set()
                for m in category_matches:
                    key = f"{m['type']}:{m['match']}"
                    if key not in seen:
                        seen.add(key)
                        unique_matches.append(m)
                matches.extend(unique_matches)
        
        # ADDITIONAL CHECK: Contextual hate patterns
        hate_pattern_matches = self._check_hate_patterns(text)
        if hate_pattern_matches:
            categories['hate_speech'] = True
            scores['hate_speech'] = max(scores['hate_speech'], 0.9)  # High severity
            matches.extend(hate_pattern_matches)
            logger.info(f"🎯 Contextual hate speech detected: {hate_pattern_matches}")
        
        return categories, scores, matches


class TextModerationService:
    def __init__(self):
        self.api_key = os.getenv('CEREBRAS_API_KEY')
        
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY must be set in environment")
        
        self.client = Cerebras(api_key=self.api_key)
        self.model = "llama-3.3-70b"
        
        # Initialize manual rules engine
        self.manual_rules = ManualRulesEngine()
        
        logger.info(f"TextModerationService initialized with Cerebras SDK + Manual Rules")
        logger.info(f"Model: {self.model}")
    
    def _get_system_prompt(self) -> str:
        return """You are an Indonesian content moderator. Analyze text for CONTEXT and INTENT.

Manual rules detected some keywords. Your job is to determine if the context is:
1. SAFE - friendly conversation, talking about pets, casual language between friends
2. REJECT - genuine harassment, threats, toxic behavior, or HATE SPEECH (SARA - ethnicity/religion/race)
3. REVIEW - borderline, unclear, or context-dependent

IMPORTANT CONTEXT RULES:
- "anjing" in "sayang anjing gw" = SAFE (talking about pet dog)
- "anjing" in "anjing lu bangsat" = REJECT (insult)
- "babi" in "daging babi" = SAFE (food)
- "babi" in "lu babi" = REJECT (insult)

CRITICAL: HATE SPEECH (SARA) DETECTION:
- ANY combination of ethnicity/religion + negative term = REJECT
- Examples: "jawa hama", "china kafir", "islam teroris", "batak kasar" = REJECT
- Even WITHOUT profanity, ethnic/religious targeting = REJECT

CATEGORIES:
- harassment = Pelecehan/intimidasi/bullying (even without bad words)
- hate_speech = Ujaran kebencian SARA (ethnicity, religion, race targeting)
- spam = Spam/iklan berlebihan

RESPONSE FORMAT (JSON):
{
  "decision": "REJECT|REVIEW|SAFE",
  "categories": {
    "harassment": true/false,
    "hate_speech": true/false,
    "spam": true/false
  },
  "scores": {
    "harassment": 0.0-1.0,
    "hate_speech": 0.0-1.0,
    "spam": 0.0-1.0
  },
  "context_notes": "brief explanation"
}

CRITICAL: If text targets ethnicity/religion/race = decision: "REJECT", hate_speech: true
If text is clearly SAFE context (pets, food, friendly), respond decision: "SAFE"
If text is clearly TOXIC (insults, threats), respond decision: "REJECT"

RESPOND ONLY WITH VALID JSON."""
    
    def _call_cerebras_api(self, text: str) -> Dict:
        """Call Cerebras API for context analysis"""
        
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    
                    # Context examples - SAFE cases (PET CONTEXT)
                    {"role": "user", "content": 'Analyze context: "gw sayang anjing gw banget"'},
                    {"role": "assistant", "content": '{"decision": "SAFE", "categories": {"harassment": false, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Expressing love for pet dog"}'},
                    
                    {"role": "user", "content": 'Analyze context: "aku sayang sama anjingku si peter"'},
                    {"role": "assistant", "content": '{"decision": "SAFE", "categories": {"harassment": false, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Talking about pet dog named Peter - clearly safe context"}'},
                    
                    {"role": "user", "content": 'Analyze context: "anjingku lucu banget"'},
                    {"role": "assistant", "content": '{"decision": "SAFE", "categories": {"harassment": false, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Describing pet dog as cute"}'},
                    
                    {"role": "user", "content": 'Analyze context: "si rocky anjing kesayangan keluarga"'},
                    {"role": "assistant", "content": '{"decision": "SAFE", "categories": {"harassment": false, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Talking about family pet dog named Rocky"}'},
                    
                    {"role": "user", "content": 'Analyze context: "anjing lu ya, tapi sayang gw sama lu wkwk"'},
                    {"role": "assistant", "content": '{"decision": "SAFE", "categories": {"harassment": false, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Friendly banter between friends"}'},
                    
                    # Context examples - REJECT cases (with profanity)
                    {"role": "user", "content": 'Analyze context: "anjing lu bangsat mati aja"'},
                    {"role": "assistant", "content": '{"decision": "REJECT", "categories": {"harassment": true, "hate_speech": false, "spam": false}, "scores": {"harassment": 1.0, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Clear toxic insult with death wish"}'},
                    
                    # Context examples - REJECT cases (hate speech WITHOUT profanity)
                    {"role": "user", "content": 'Analyze context: "jawa hama"'},
                    {"role": "assistant", "content": '{"decision": "REJECT", "categories": {"harassment": false, "hate_speech": true, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 1.0, "spam": 0.0}, "context_notes": "Ethnic hate speech: calling Javanese people pests"}'},
                    
                    {"role": "user", "content": 'Analyze context: "china kafir semua"'},
                    {"role": "assistant", "content": '{"decision": "REJECT", "categories": {"harassment": false, "hate_speech": true, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 1.0, "spam": 0.0}, "context_notes": "Religious hate speech targeting Chinese ethnicity"}'},
                    
                    {"role": "user", "content": 'Analyze context: "batak kasar semua"'},
                    {"role": "assistant", "content": '{"decision": "REJECT", "categories": {"harassment": false, "hate_speech": true, "spam": false}, "scores": {"harassment": 0.0, "hate_speech": 1.0, "spam": 0.0}, "context_notes": "Ethnic stereotyping and derogatory generalization"}'},
                    
                    {"role": "user", "content": 'Analyze context: "kamu selalu salah, gak ada yang bener, mending keluar aja dari sini"'},
                    {"role": "assistant", "content": '{"decision": "REJECT", "categories": {"harassment": true, "hate_speech": false, "spam": false}, "scores": {"harassment": 0.9, "hate_speech": 0.0, "spam": 0.0}, "context_notes": "Clear harassment pattern - belittling and exclusion"}'},
                    
                    # Actual query
                    {"role": "user", "content": f'Analyze context: "{text}"'}
                ],
                model=self.model,
                max_completion_tokens=300,
                temperature=0.1,
                top_p=1,
                stream=False
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Clean JSON
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON
            import json
            result = json.loads(response_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Cerebras API error: {e}")
            # Return safe defaults if AI fails
            return {
                "decision": "SAFE",
                "categories": {"harassment": False, "hate_speech": False, "spam": False},
                "scores": {"harassment": 0.0, "hate_speech": 0.0, "spam": 0.0},
                "context_notes": f"AI error: {str(e)}"
            }
    
    def moderate(self, text: str, mode: str = 'rules', lang: str = 'id') -> ModerationResult:
        """Moderate text using HYBRID approach: Manual Rules + AI Context"""
        
        if not text or not text.strip():
            return ModerationResult(
                status=ModerationStatus.SUCCESS,
                decision=ModerationDecision.AUTO_APPROVE,
                flagged=False,
                categories={},
                scores={},
                matches=[],
                original_lang=lang
            )
        
        try:
            # STEP 1: Manual keyword detection (ALWAYS runs)
            manual_categories, manual_scores, manual_matches = self.manual_rules.check_text(text)
            
            logger.info(f"📋 Manual Detection: {sum(manual_categories.values())} categories flagged")
            if manual_matches:
                match_summary = [f"{m['original']}→{m['match']}" for m in manual_matches[:5]]
                logger.info(f"   Matches: {match_summary}")
            else:
                logger.info(f"   Text analyzed: '{text[:50]}...'")
                logger.info(f"   No manual matches found")
            
            # STEP 2: AI context analysis (runs in parallel)
            ai_result = self._call_cerebras_api(text)
            ai_categories = ai_result.get('categories', {})
            ai_scores = ai_result.get('scores', {})
            
            logger.info(f"🤖 AI Analysis: {ai_result.get('context_notes', 'N/A')}")
            
            # STEP 3: Merge results (Manual takes priority for keywords)
            final_categories = {**manual_categories, **ai_categories}
            final_scores = {**manual_scores, **ai_scores}
            final_matches = manual_matches.copy()
            
            # STEP 4: Make decision with STRICT rules for hate speech
            # Define severity levels
            critical_categories = ['sexual', 'violence', 'hate_speech']  # Always REJECT, no AI override
            strict_profanity = ['profanity']  # Strict profanity - mostly REJECT, AI can only upgrade severity
            contextual_categories = ['offensive', 'harassment', 'spam']  # AI decides based on context
            
            has_critical_violation = any(final_categories.get(cat, False) for cat in critical_categories)
            has_strict_profanity = final_categories.get('profanity', False)
            has_contextual_violation = any(final_categories.get(cat, False) for cat in contextual_categories)
            
            ai_decision = ai_result.get('decision', 'SAFE').upper()
            ai_says_safe = ai_decision == 'SAFE'
            ai_says_reject = ai_decision == 'REJECT'
            
            # Check if profanity detected is from truly ambiguous words (can be context-dependent)
            # These words CAN be innocent when talking about pets, animals, or food
            ambiguous_words = ['anjing', 'babi', 'monyet', 'asu', 'kucing']  # Words that CAN be innocent in right context
            ambiguous_partials = ['jing', 'njing', 'ajg', 'njir', 'anjir']  # Partial words that might be from ambiguous words
            
            detected_words = [m['original'].lower() for m in manual_matches if m['type'] == 'profanity']
            
            # Check if all detected words are either ambiguous or ambiguous partials
            def is_ambiguous_or_partial(word):
                if word in ambiguous_words:
                    return True
                # Check if it's a partial of an ambiguous word
                if word in ambiguous_partials:
                    return True
                # Check if any ambiguous word contains this word as substring
                for amb_word in ambiguous_words:
                    if word in amb_word and len(word) >= 3:  # At least 3 chars to avoid false matches
                        return True
                return False
            
            is_truly_ambiguous = all(is_ambiguous_or_partial(word) for word in detected_words)
            
            # Additional check: Look for pet/animal context indicators in the text
            # PET CONTEXT indicators (positive ownership/affection)
            pet_indicators = [
                'sayang', 'pelihara', 'lucu', 'imut', 'kesayangan', 
                'nama', 'si ', 'milik', 'punya', 'ku ', 'gw ', 'aku ',
                'keluarga', 'rumah', 'anak', 'anakan', 'adopsi',
                'manis', 'gemesin', 'menggemaskan', 'jinak', 'nurut',
                'vaksin', 'dokter hewan', 'kandang', 'makanan anjing',
                'jalan-jalan', 'main', 'lari-lari'
            ]
            
            # FOOD CONTEXT indicators
            food_indicators = ['daging', 'masak', 'goreng', 'bakar', 'rebus', 'makan', 'enak']
            
            # INSULT CONTEXT indicators (negative targeting of person)
            insult_indicators = [
                'lu ', 'kamu ', 'elu ', 'loe ', 'kalian ', 'dia ',
                'lo ', 'kau ', 'engkau ', 'ente '
            ]
            
            # INSULT INTENSIFIERS (words that make it clearly an insult)
            insult_intensifiers = [
                'bangsat', 'tolol', 'bego', 'goblok', 'bodoh',
                'tai', 'sampah', 'busuk', 'jelek', 'brengsek'
            ]
            
            text_lower = text.lower()
            has_pet_context = any(indicator in text_lower for indicator in pet_indicators)
            has_food_context = any(indicator in text_lower for indicator in food_indicators)
            has_insult_target = any(indicator in text_lower for indicator in insult_indicators)
            has_insult_intensifier = any(indicator in text_lower for indicator in insult_intensifiers)
            
            # SMART CONTEXT DETECTION:
            # If text has "anjing/babi/etc" + person pronoun (lu/kamu) BUT NO pet indicators
            # → Likely an insult, not talking about pet
            is_likely_insult = (
                has_insult_target and 
                not has_pet_context and 
                not has_food_context
            )
            
            # If has explicit insult words alongside → DEFINITELY insult
            is_definite_insult = has_insult_intensifier
            
            # Combined safe context: pet/food context AND NOT likely insult
            has_safe_context = (has_pet_context or has_food_context) and not is_definite_insult
            
            # Decision logic (PRIORITY ORDER):
            # 1. Critical violations (sexual/violence/HATE SPEECH) → ALWAYS REJECT
            # 2. Ambiguous + safe context (pet/food) → APPROVE
            # 3. Ambiguous + likely insult context (person targeting) → REJECT (NEW!)
            # 4. Explicit non-ambiguous profanity → ALWAYS REJECT
            # 5. Ambiguous + unclear context → REVIEW
            # 6. AI says REJECT → REJECT
            # 7. Contextual violations → REVIEW
            
            if has_critical_violation:
                # Sexual, violence, HATE SPEECH → ALWAYS REJECT
                decision = ModerationDecision.AUTO_REJECT
                flagged = True
                logger.info(f"🚫 REJECT: Critical violation detected ({critical_categories})")
                
            elif has_strict_profanity and is_truly_ambiguous and (ai_says_safe or has_safe_context):
                # Ambiguous words (anjing, babi) in safe context (pet, food, or AI confirms) → APPROVE
                decision = ModerationDecision.AUTO_APPROVE
                flagged = False
                # Clear the ambiguous flags
                final_categories['profanity'] = False
                final_scores['profanity'] = 0.0
                logger.info(f"✅ APPROVE: Ambiguous words in safe context: {detected_words} (pet/food context detected or AI confirmed safe)")
                
            elif has_strict_profanity and is_truly_ambiguous and (is_likely_insult or is_definite_insult or ai_says_reject):
                # Ambiguous words BUT in clear insult context → REJECT
                decision = ModerationDecision.AUTO_REJECT
                flagged = True
                reason = "definite insult" if is_definite_insult else "likely insult context (person targeting)" if is_likely_insult else "AI detected insult"
                logger.info(f"🚫 REJECT: Ambiguous words used as insult: {detected_words} ({reason})")
                
            elif has_strict_profanity and not is_truly_ambiguous:
                # Explicit profanity (memek, kontol, tempek, etc) that is NOT ambiguous → ALWAYS REJECT
                decision = ModerationDecision.AUTO_REJECT
                flagged = True
                logger.info(f"🚫 REJECT: Explicit profanity detected: {detected_words}")
                
            elif has_strict_profanity and is_truly_ambiguous:
                # Ambiguous words, but NO safe context and AI unclear → REVIEW
                decision = ModerationDecision.REVIEW_REQUIRED
                flagged = True
                logger.info(f"⚠️ REVIEW: Ambiguous words need human review: {detected_words}")
                
            elif ai_says_reject:
                # AI detected context-based violation (harassment without explicit words)
                decision = ModerationDecision.AUTO_REJECT
                flagged = True
                logger.info(f"🚫 REJECT: AI detected toxic context")
                
            elif has_contextual_violation:
                # Offensive/harassment/spam → depends on severity
                decision = ModerationDecision.REVIEW_REQUIRED
                flagged = True
                logger.info(f"⚠️ REVIEW: Contextual violation detected")
                
            else:
                # Clean text
                decision = ModerationDecision.AUTO_APPROVE
                flagged = False
                logger.info(f"✅ APPROVE: Clean text")
            
            logger.info(f"✅ Final: {decision.value} | Categories: {final_categories}")
            
            return ModerationResult(
                status=ModerationStatus.SUCCESS,
                decision=decision,
                flagged=flagged,
                categories=final_categories,
                scores=final_scores,
                matches=final_matches,
                raw_response={
                    'manual': {'categories': manual_categories, 'scores': manual_scores},
                    'ai': ai_result
                },
                original_lang=lang,
                message=ai_result.get('context_notes')
            )
            
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            
            return ModerationResult(
                status=ModerationStatus.ERROR,
                decision=ModerationDecision.REVIEW_REQUIRED,
                flagged=True,
                categories={},
                scores={},
                matches=[],
                message=f'Error: {str(e)}',
                original_lang=lang
            )
    
    def moderate_username(self, username: str, lang: str = 'id') -> ModerationResult:
        return self.moderate(username, mode='username', lang=lang)


def moderate_text(text: str, lang: str = 'id', mode: str = 'rules') -> Dict:
    """Main entry point for moderation"""
    service = TextModerationService()
    result = service.moderate(text, mode=mode, lang=lang)
    return result.to_dict()