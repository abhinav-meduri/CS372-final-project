"""
Patent Data Preprocessor
Merges claims and summary data into unified JSONL format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
from collections import defaultdict

from .loader import PatentDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentPreprocessor:
    """Preprocess patent data into unified JSONL format."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.loader = PatentDataLoader(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_claims_df(self, claims_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Process claims DataFrame into grouped structure per patent.
        
        Returns:
            Dict mapping patent_id to claims data
        """
        patent_claims = defaultdict(lambda: {"claims": [], "independent_claims": []})
        
        for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc="Processing claims"):
            patent_id = str(row['patent_id'])
            claim_text = str(row['claim_text']) if pd.notna(row['claim_text']) else ""
            claim_num = int(row['claim_number']) if pd.notna(row['claim_number']) else 0
            is_dependent = pd.notna(row['dependent'])  # NaN means independent
            
            claim_entry = {
                "claim_num": claim_num,
                "text": claim_text,
                "dependent": is_dependent
            }
            
            patent_claims[patent_id]["claims"].append(claim_entry)
            
            # Track independent claims separately
            if not is_dependent:
                patent_claims[patent_id]["independent_claims"].append(claim_text)
        
        return dict(patent_claims)
    
    def process_summary_df(self, summary_df: pd.DataFrame) -> Dict[str, str]:
        """
        Process summary DataFrame into dict mapping patent_id to summary.
        """
        summaries = {}
        for _, row in tqdm(summary_df.iterrows(), total=len(summary_df), desc="Processing summaries"):
            patent_id = str(row['patent_id'])
            summary_text = str(row['summary_text']) if pd.notna(row['summary_text']) else ""
            summaries[patent_id] = summary_text
        return summaries
    
    def merge_patent_data(
        self, 
        claims_data: Dict[str, dict], 
        summaries: Dict[str, str],
        year: int
    ) -> List[dict]:
        """
        Merge claims and summaries into unified patent records.
        """
        patents = []
        
        # Get union of all patent IDs
        all_patent_ids = set(claims_data.keys()) | set(summaries.keys())
        
        for patent_id in tqdm(all_patent_ids, desc=f"Merging year {year}"):
            claims_info = claims_data.get(patent_id, {"claims": [], "independent_claims": []})
            summary = summaries.get(patent_id, "")
            
            # Sort claims by claim number
            claims = sorted(claims_info["claims"], key=lambda x: x["claim_num"])
            independent_claims = claims_info["independent_claims"]
            
            # Extract title from first claim or summary (heuristic)
            title = self._extract_title(summary, claims)
            
            patent = {
                "patent_id": patent_id,
                "title": title,
                "abstract": summary[:2000] if summary else "",  # Truncate long summaries
                "summary": summary,
                "claims": claims,
                "independent_claims": independent_claims[:5],  # Keep top 5 independent claims
                "num_claims": len(claims),
                "year": year
            }
            
            patents.append(patent)
        
        return patents
    
    def _extract_title(self, summary: str, claims: List[dict]) -> str:
        """
        Extract a title from the summary or first claim.
        This is a heuristic since PatentsView summary doesn't have a dedicated title field.
        """
        # Common section headers to ignore (not real titles)
        SECTION_HEADERS = {
            'background', 'background of the invention', 'field of the invention',
            'technical field', 'field of invention', 'summary', 'summary of the invention',
            'brief description of the drawings', 'detailed description',
            'cross-reference to related applications', 'cross-reference to prior application',
            'related applications', 'priority claim', 'abstract', 'claims',
            'description of the related art', 'description of related art',
            'brief description', 'government rights', 'federally sponsored research'
        }
        
        # Check if this is a design patent (they have proper titles)
        if summary and 'ornamental design' in summary.lower():
            # Design patent - the title IS the summary
            return summary.strip()
        
        if summary:
            # Try to find a title-like string in the first few lines
            lines = summary.split('\n')
            for line in lines[:10]:
                line = line.strip()
                line_lower = line.lower()
                
                # Skip section headers
                if line_lower in SECTION_HEADERS:
                    continue
                if any(line_lower.startswith(h) for h in SECTION_HEADERS):
                    continue
                
                # Skip empty or very short lines
                if len(line) < 10:
                    continue
                    
                # Skip lines that look like section headers
                if line.isupper() and len(line) < 50:
                    continue
                
                # Look for substantive content
                if line and len(line) < 200:
                    # If the line contains technical keywords, it might be title-like
                    if any(word in line_lower for word in ['method', 'system', 'apparatus', 'device', 'composition', 'process', 'article', 'compound']):
                        return line[:200]
                    # If line starts with a capitalized word and is sentence-like
                    if line[0].isupper() and ' ' in line and not line.endswith(':'):
                        return line[:200]
        
        # Fallback: use first independent claim summary
        if claims:
            first_claim = claims[0].get('text', '')
            if first_claim:
                # Take first sentence or first 150 chars
                end_idx = min(
                    first_claim.find('.') + 1 if '.' in first_claim else len(first_claim),
                    150
                )
                return first_claim[:end_idx].strip()
        
        return "Patent Application"
    
    def process_year(self, year: int) -> List[dict]:
        """Process all patent data for a single year."""
        logger.info(f"Processing year {year}...")
        
        # Load data
        claims_df = self.loader.load_claims(year)
        summary_df = self.loader.load_summary(year)
        
        logger.info(f"  Loaded {len(claims_df)} claims, {len(summary_df)} summaries")
        
        # Process into structured format
        claims_data = self.process_claims_df(claims_df)
        summaries = self.process_summary_df(summary_df)
        
        # Merge
        patents = self.merge_patent_data(claims_data, summaries, year)
        
        logger.info(f"  Created {len(patents)} patent records for year {year}")
        
        return patents
    
    def process_all_years(
        self, 
        years: Optional[List[int]] = None,
        output_file: str = "patents.jsonl"
    ) -> str:
        """
        Process all years and save to JSONL.
        
        Returns:
            Path to output file
        """
        if years is None:
            years = self.loader.get_available_years()
        
        logger.info(f"Processing years: {years}")
        
        all_patents = []
        for year in years:
            try:
                patents = self.process_year(year)
                all_patents.extend(patents)
            except Exception as e:
                logger.error(f"Error processing year {year}: {e}")
                continue
        
        # Save to JSONL
        output_path = self.output_dir / output_file
        logger.info(f"Saving {len(all_patents)} patents to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for patent in tqdm(all_patents, desc="Writing JSONL"):
                f.write(json.dumps(patent, ensure_ascii=False) + '\n')
        
        # Save statistics
        stats = {
            "total_patents": len(all_patents),
            "years_processed": years,
            "patents_per_year": {year: sum(1 for p in all_patents if p['year'] == year) for year in years}
        }
        stats_path = self.output_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing complete! Stats: {stats}")
        
        return str(output_path)
    
    def create_train_val_test_split(
        self,
        patents_file: str = "patents.jsonl",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        """
        Create train/val/test splits by patent ID.
        
        This ensures no leakage - all claims from a patent are in the same split.
        """
        np.random.seed(seed)
        
        # Load patent IDs
        patents_path = self.output_dir / patents_file
        patent_ids = []
        
        with open(patents_path, 'r') as f:
            for line in f:
                patent = json.loads(line)
                patent_ids.append(patent['patent_id'])
        
        # Shuffle and split
        patent_ids = np.array(patent_ids)
        np.random.shuffle(patent_ids)
        
        n_total = len(patent_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_ids = patent_ids[:n_train].tolist()
        val_ids = patent_ids[n_train:n_train + n_val].tolist()
        test_ids = patent_ids[n_train + n_val:].tolist()
        
        # Save split files
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            path = self.output_dir / f"{name}_ids.txt"
            with open(path, 'w') as f:
                f.write('\n'.join(ids))
            logger.info(f"Saved {len(ids)} {name} IDs to {path}")
        
        return {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids)
        }


def process_sample(n_rows: int = 1000):
    """Process a small sample for testing."""
    preprocessor = PatentPreprocessor()
    loader = preprocessor.loader
    
    # Load small sample
    year = 2021
    claims_sample = loader.peek_columns(year, "claims", nrows=n_rows)
    summary_sample = loader.peek_columns(year, "summary", nrows=n_rows)
    
    # Process
    claims_data = preprocessor.process_claims_df(claims_sample)
    summaries = preprocessor.process_summary_df(summary_sample)
    patents = preprocessor.merge_patent_data(claims_data, summaries, year)
    
    print(f"\nSample processed: {len(patents)} patents")
    
    # Show sample
    if patents:
        sample = patents[0]
        print(f"\nSample patent:")
        print(f"  ID: {sample['patent_id']}")
        print(f"  Title: {sample['title'][:100]}...")
        print(f"  Num claims: {sample['num_claims']}")
        print(f"  Num independent: {len(sample['independent_claims'])}")
        if sample['claims']:
            print(f"  First claim (truncated): {sample['claims'][0]['text'][:200]}...")
    
    return patents


def main():
    """Main entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess patent data")
    parser.add_argument("--years", nargs="+", type=int, help="Years to process")
    parser.add_argument("--sample", action="store_true", help="Process small sample only")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size")
    parser.add_argument("--output", default="patents.jsonl", help="Output filename")
    parser.add_argument("--split", action="store_true", help="Create train/val/test split")
    
    args = parser.parse_args()
    
    if args.sample:
        process_sample(args.sample_size)
    else:
        preprocessor = PatentPreprocessor()
        preprocessor.process_all_years(years=args.years, output_file=args.output)
        
        if args.split:
            preprocessor.create_train_val_test_split(args.output)


if __name__ == "__main__":
    main()




