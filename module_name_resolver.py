#!/usr/bin/env python3
"""
Module Name Resolver
--------------------
Unified module for extracting/resolving NLC module names from PDFs.

Supports two strategies:
1. Filename-priority: Uses PDF filename with smart cleanup and manual overrides (recommended)
2. Content-based: Extracts name from PDF text content with DB matching

Usage:
    from module_name_resolver import ModuleNameResolver
    
    resolver = ModuleNameResolver(
        modules_db_path="nlc_modules.json",
        overrides_path="module_overrides.json"
    )
    
    # Filename-priority (recommended)
    name, method = resolver.resolve(pdf_path, strategy="filename")
    
    # Content-based
    name, method = resolver.resolve(pdf_path, strategy="content")
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

# Optional dependency - only needed for content-based detection
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class ModuleNameResolver:
    """
    Resolves NLC module names from PDF files using multiple strategies.
    """
    
    def __init__(
        self,
        modules_db_path: Optional[Path] = None,
        overrides_path: Optional[Path] = None
    ):
        """
        Initialize the resolver.
        
        Args:
            modules_db_path: Path to nlc_modules.json (canonical names database)
            overrides_path: Path to module_overrides.json (filename → name mappings)
        """
        self.db = self._load_modules_db(modules_db_path)
        self.overrides = self._load_overrides(overrides_path)
        
        # Precompute normalized canonical names for faster matching
        self._normalized_canonicals = {
            self._normalize(name): name 
            for name in self.db.get("canonical_names", [])
        }
        
        # Precompute normalized aliases
        self._normalized_aliases = {
            alias: canon
            for alias, canon in self.db.get("aliases", {}).items()
        }
    
    def _load_modules_db(self, path: Optional[Path]) -> dict:
        """Load the canonical module names database."""
        if path is None:
            script_dir = Path(__file__).parent
            path = script_dir / "nlc_modules.json"
        else:
            path = Path(path)
        
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {"canonical_names": [], "aliases": {}}
    
    def _load_overrides(self, path: Optional[Path]) -> dict:
        """Load manual filename → module name overrides."""
        if path is None:
            script_dir = Path(__file__).parent
            path = script_dir / "module_overrides.json"
        else:
            path = Path(path)
        
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("overrides", {})
        return {}
    
    @staticmethod
    def _normalize(s: str) -> str:
        """Normalize a string for comparison."""
        s = s.lower()
        s = re.sub(r"[''`]", "", s)  # Remove apostrophes
        s = re.sub(r"[^a-z0-9\s]", " ", s)  # Replace punctuation with space
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    def resolve(self, pdf_path: Path, strategy: str = "filename") -> tuple[str, str]:
        """
        Resolve the module name for a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            strategy: "filename" (recommended) or "content"
        
        Returns:
            Tuple of (module_name, detection_method)
        """
        if strategy == "filename":
            return self.resolve_from_filename(pdf_path)
        elif strategy == "content":
            return self.resolve_from_content(pdf_path)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    # =========================================================================
    # FILENAME-BASED RESOLUTION (recommended)
    # =========================================================================
    
    def resolve_from_filename(self, pdf_path: Path) -> tuple[str, str]:
        """
        Resolve module name from filename with overrides.
        
        Priority:
        1. Check manual overrides (exact filename match)
        2. Clean filename and match against canonical DB
        3. Return cleaned filename
        """
        stem = pdf_path.stem
        
        # 1. Check exact override
        if stem in self.overrides:
            return self.overrides[stem], "filename_override"
        
        # Check with common variations removed (e.g., " (1)" suffix)
        stem_base = re.sub(r'\s*\(\d+\)$', '', stem)
        if stem_base != stem and stem_base in self.overrides:
            return self.overrides[stem_base], "filename_override"
        
        # 2. Clean filename and try DB match
        cleaned = self._clean_filename(stem)
        
        # Try to match against canonical names
        matched = self._match_against_db(cleaned)
        if matched != cleaned:
            return matched, "filename_db_match"
        
        # 3. Return cleaned filename
        return cleaned, "filename_cleaned"
    
    def _clean_filename(self, raw_name: str) -> str:
        """
        Clean a PDF filename to extract the module name.
        
        Handles:
        - URL encoding (+, %20, etc.)
        - Common suffixes (build, bom, etc.)
        - Version markers
        - Proper capitalization
        """
        name = raw_name
        
        # URL decode
        name = unquote(name)
        name = name.replace("+", " ")
        name = name.replace("_", " ")
        name = re.sub(r'-+', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove common prefixes
        prefixes = [
            r'^nonlinearcircuits\s*',
            r'^nlc\s*[-–—]\s*',
            r'^nlc\s+',
        ]
        for prefix in prefixes:
            name = re.sub(prefix, '', name, flags=re.IGNORECASE)
        
        # Remove common suffixes (multiple passes)
        for _ in range(3):
            old_name = name
            suffixes = [
                r'\s+new\s*$',
                r'\s+build\s+and\s+bom\s*$',
                r'\s+build\s*&\s*bom\s*$',
                r'\s+build\s+bom\s*$',
                r'\s+bom\s+build\s*$',
                r'\s+build\s*$',
                r'\s+bom\s*$',
                r'\s+vers?\.?\s*\d*\s*$',
                r'\s*\(\d+\)\s*$',
                r'\s+pcb\s*$',
            ]
            for suffix in suffixes:
                name = re.sub(suffix, '', name, flags=re.IGNORECASE)
            if name == old_name:
                break
        
        # Remove trailing version numbers (but not module version numbers like "Delay No More 3")
        # Only remove single trailing digits that look like file versioning
        name_without_trailing = re.sub(r'\s+\d\s*$', '', name)
        
        # Check if the version IS part of the module name
        if self._normalize(name) in self._normalized_canonicals:
            pass  # Keep the number
        elif self._normalize(name_without_trailing) in self._normalized_canonicals:
            name = name_without_trailing
        else:
            # Neither matches, remove trailing single digit
            name = name_without_trailing
        
        # Clean up
        name = re.sub(r'\s+', ' ', name).strip().strip('-_ ')
        
        # Apply proper capitalization
        name = self._apply_capitalization(name)
        
        return name if name else raw_name
    
    def _apply_capitalization(self, name: str) -> str:
        """Apply proper capitalization, preserving known acronyms."""
        acronyms = {
            'vco': 'VCO', 'vca': 'VCA', 'vcf': 'VCF', 'lfo': 'LFO',
            'adsr': 'ADSR', 'lpg': 'LPG', 'ota': 'OTA', 'cem': 'CEM',
            'cmos': 'CMOS', 'diy': 'DIY', 'cv': 'CV', 'fm': 'FM',
            'hp': 'HP', '1u': '1U', '4hp': '4HP', '8hp': '8HP',
            'mbd': 'MBD', 'xor': 'XOR', 'wamod': 'WAMOD', 'pill': 'PiLL',
            'mun': 'MUN', 'bom': 'BOM', 'pcb': 'PCB',
        }
        
        words = name.split()
        result = []
        
        for word in words:
            lower = word.lower()
            if lower in acronyms:
                result.append(acronyms[lower])
            elif word.isupper() and len(word) > 1:
                # Preserve existing uppercase
                result.append(word)
            else:
                result.append(word.capitalize())
        
        return ' '.join(result)
    
    def _match_against_db(self, name: str) -> str:
        """Try to match a name against the canonical database."""
        if not self.db.get("canonical_names"):
            return name
        
        normalized = self._normalize(name)
        
        # 1. Check aliases
        if normalized in self._normalized_aliases:
            return self._normalized_aliases[normalized]
        
        # 2. Exact match against canonical names
        if normalized in self._normalized_canonicals:
            return self._normalized_canonicals[normalized]
        
        # 3. Prefix/contains matching (prefer longer matches)
        potential_matches = []
        for norm_canon, canon in self._normalized_canonicals.items():
            if len(normalized) >= 4 and len(norm_canon) >= 4:
                if norm_canon == normalized:
                    potential_matches.append((canon, 1000 + len(norm_canon)))
                elif normalized.startswith(norm_canon + " "):
                    potential_matches.append((canon, 100 + len(norm_canon)))
                elif norm_canon.startswith(normalized + " "):
                    potential_matches.append((canon, 100 + len(norm_canon)))
                elif norm_canon in normalized:
                    potential_matches.append((canon, len(norm_canon)))
                elif normalized in norm_canon:
                    potential_matches.append((canon, len(norm_canon)))
        
        if potential_matches:
            potential_matches.sort(key=lambda x: -x[1])
            return potential_matches[0][0]
        
        return name
    
    # =========================================================================
    # CONTENT-BASED RESOLUTION
    # =========================================================================
    
    def resolve_from_content(self, pdf_path: Path) -> tuple[str, str]:
        """
        Extract module name from PDF content.
        
        Strategy order:
        1. Exact match against canonical names in text
        2. Alias match
        3. Title extraction from first lines
        4. Word frequency analysis
        5. Fallback to filename
        """
        if not HAS_PDFPLUMBER:
            # Fall back to filename if pdfplumber not available
            return self.resolve_from_filename(pdf_path)
        
        text = self._extract_text(pdf_path)
        
        if not text.strip():
            name, _ = self.resolve_from_filename(pdf_path)
            return name, "filename_fallback"
        
        normalized_text = self._normalize(text)
        
        # 1. Direct match against canonical names (longest first)
        sorted_names = sorted(
            self.db.get("canonical_names", []),
            key=len, reverse=True
        )
        
        for canon in sorted_names:
            norm_canon = self._normalize(canon)
            pattern = r'\b' + re.escape(norm_canon) + r'\b'
            if re.search(pattern, normalized_text):
                return canon, "content_exact_match"
        
        # 2. Check aliases
        for alias, canon in self._normalized_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, normalized_text):
                return canon, "content_alias_match"
        
        # 3. Title extraction
        lines = text.strip().split('\n')
        title = self._extract_title_from_lines(lines)
        if title:
            return title, "content_title_extraction"
        
        # 4. Word frequency analysis
        freq_name = self._extract_from_frequency(text)
        if freq_name:
            return freq_name, "content_frequency_analysis"
        
        # 5. Fallback to filename
        name, _ = self.resolve_from_filename(pdf_path)
        return name, "filename_fallback"
    
    def _extract_text(self, pdf_path: Path, max_pages: int = 3) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                text_parts = []
                for page in pdf.pages[:max_pages]:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n".join(text_parts)
        except Exception as e:
            print(f"  Warning: Could not extract text from {pdf_path.name}: {e}")
            return ""
    
    def _extract_title_from_lines(self, lines: list[str]) -> Optional[str]:
        """Look for module name in the first few lines."""
        skip_patterns = [
            r'^(qty|quantity|value|component|part|ref|designator)',
            r'^(mouser|tayda|digikey)',
            r'^\d+$',
            r'^[A-Z]\d+$',
            r'^page\s*\d+',
            r'^bom\s*$',
            r'^bill\s+of\s+materials',
        ]
        
        checked = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            checked += 1
            if checked > 10:
                break
            
            line_lower = line.lower()
            if any(re.match(pat, line_lower) for pat in skip_patterns):
                continue
            
            normalized_line = self._normalize(line)
            
            # Check against canonical names
            for norm_canon, canon in self._normalized_canonicals.items():
                if normalized_line == norm_canon:
                    return canon
                if normalized_line.startswith(norm_canon + " "):
                    return canon
                if norm_canon in normalized_line and len(normalized_line) < len(norm_canon) * 2:
                    return canon
            
            # Try matching short lines
            if 3 < len(line) < 50:
                matched = self._match_against_db(line)
                if matched != line:
                    return matched
        
        return None
    
    def _extract_from_frequency(self, text: str) -> Optional[str]:
        """Use word frequency to identify module name."""
        stop_words = {
            'qty', 'quantity', 'value', 'component', 'part', 'ref', 'reference',
            'designator', 'description', 'footprint', 'package', 'mouser', 'digikey',
            'tayda', 'notes', 'comments', 'bom', 'bill', 'materials', 'page',
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are', 'not',
            'use', 'using', 'used', 'can', 'will', 'should', 'may', 'all', 'any',
            'resistor', 'resistors', 'capacitor', 'capacitors', 'diode', 'diodes',
            'transistor', 'transistors', 'led', 'leds', 'socket', 'sockets',
            'jack', 'jacks', 'pot', 'pots', 'potentiometer', 'knob', 'knobs',
            'pin', 'pins', 'header', 'headers', 'connector', 'connectors',
            'eurorack', 'power', 'supply', 'ground', 'gnd', 'vcc', 'vee',
            'ohm', 'ohms', 'farad', 'nf', 'uf', 'pf', 'mf',
            'smd', 'thru', 'hole', 'through', 'smt',
            'build', 'guide', 'instructions', 'assembly',
        }
        
        normalized_text = self._normalize(text)
        words = normalized_text.split()
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Generate n-grams
        def get_ngrams(word_list, n):
            return [' '.join(word_list[i:i+n]) for i in range(len(word_list)-n+1)]
        
        all_ngrams = []
        for n in range(2, 5):
            all_ngrams.extend(get_ngrams(filtered, n))
        all_ngrams.extend(filtered)
        
        ngram_counts = Counter(all_ngrams)
        
        # Match against canonical names
        for ngram, count in ngram_counts.most_common(50):
            if count < 2:
                continue
            
            for norm_canon, canon in self._normalized_canonicals.items():
                if ngram == norm_canon:
                    return canon
                if norm_canon in ngram or ngram in norm_canon:
                    overlap = len(set(ngram.split()) & set(norm_canon.split()))
                    if overlap >= min(2, len(norm_canon.split())):
                        return canon
        
        return None


# =============================================================================
# STANDALONE FUNCTIONS (for backward compatibility)
# =============================================================================

_default_resolver: Optional[ModuleNameResolver] = None


def get_default_resolver() -> ModuleNameResolver:
    """Get or create the default resolver instance."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = ModuleNameResolver()
    return _default_resolver


def load_module_names_db(json_path: Path = None) -> dict:
    """Load the canonical module names database. (Backward compatible)"""
    resolver = ModuleNameResolver(modules_db_path=json_path)
    return resolver.db


def clean_module_name(raw_name: str, db: dict = None) -> str:
    """Clean a module name from filename. (Backward compatible)"""
    resolver = get_default_resolver()
    # Create a fake path just for the stem
    cleaned = resolver._clean_filename(raw_name)
    if db:
        return resolver._match_against_db(cleaned)
    return cleaned


def clean_module_name_from_filename(raw_name: str, db: dict = None) -> str:
    """Alias for clean_module_name. (Backward compatible)"""
    return clean_module_name(raw_name, db)


def match_module_name(cleaned_name: str, db: dict) -> str:
    """Match against canonical names. (Backward compatible)"""
    resolver = get_default_resolver()
    return resolver._match_against_db(cleaned_name)


def extract_module_name_from_content(pdf_path: Path, db: dict) -> tuple[str, str]:
    """Extract module name from PDF content. (Backward compatible)"""
    resolver = get_default_resolver()
    return resolver.resolve_from_content(pdf_path)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("Module Name Resolver Test")
    print("=" * 60)
    
    resolver = ModuleNameResolver()
    
    if resolver.overrides:
        print(f"Loaded {len(resolver.overrides)} filename overrides")
    if resolver.db.get("canonical_names"):
        print(f"Loaded {len(resolver.db['canonical_names'])} canonical names")
    print()
    
    # Test filenames
    test_files = [
        "MBD+BOM+and+build.pdf",
        "LET'S+GET+FENESTRATED+Build+and+BOM.pdf",
        "4hp_mix_build_and_bom.pdf",
        "statues_build_and_bom.pdf",
        "Clump+Build+and+BOM.pdf",
        "1U+MULCHO+build+and+BOM.pdf",
        "Hyperchaos+Deluxe+build+and+BOM.pdf",
        "sloth_chaos2.pdf",
        "DUAL+LPG+build+and+BOM.pdf",
        "noiro-ze_vcf_vca_build_and_bom.pdf",
    ]
    
    print("Filename → Module Name (filename strategy)")
    print("-" * 60)
    
    for filename in test_files:
        fake_path = Path(filename)
        name, method = resolver.resolve(fake_path, strategy="filename")
        indicator = "✓" if method == "filename_override" else "○"
        print(f"{indicator} {filename}")
        print(f"  → {name} ({method})")
        print()
    
    # If PDFs provided as args, test them
    if len(sys.argv) > 1:
        print("\nTesting actual PDF files:")
        print("-" * 60)
        for pdf_arg in sys.argv[1:]:
            pdf_path = Path(pdf_arg)
            if pdf_path.exists():
                name_f, method_f = resolver.resolve(pdf_path, strategy="filename")
                name_c, method_c = resolver.resolve(pdf_path, strategy="content")
                print(f"\n{pdf_path.name}:")
                print(f"  Filename strategy: {name_f} ({method_f})")
                print(f"  Content strategy:  {name_c} ({method_c})")