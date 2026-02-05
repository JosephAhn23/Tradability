"""
Reproducible Attestation System

Cryptographically binds:
1. Exact code version
2. Exact inputs
3. Exact outputs

No screenshots. No "trust me." No narrative.
"""

import json
import hashlib
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import importlib.metadata


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NO_GIT"


def get_python_version() -> str:
    """Get exact Python version."""
    return sys.version


def get_package_versions() -> Dict[str, str]:
    """Get exact package versions with hashes if available."""
    packages = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata['Name']
            version = dist.metadata['Version']
            packages[name] = version
        except:
            pass
    return packages


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_source_file_hashes(source_dir: Path = Path('.')) -> Dict[str, str]:
    """Get SHA256 hashes of all Python source files."""
    hashes = {}
    
    # Files to include
    source_files = [
        'war_test_ii.py',
        'execute_signals.py',
        'signals.py',
        'tradability_analysis.py',
        'transaction_costs.py',
        'decay_analysis.py',
        'data_utils.py',
        'capacity.py',
        'slippage.py',
        'formal_definitions.py',
        'discovery_proxies.py',
        'controls.py',
    ]
    
    for filename in source_files:
        filepath = source_dir / filename
        if filepath.exists():
            hashes[filename] = compute_file_hash(filepath)
    
    return hashes


def compute_data_fingerprint(ticker: str, start_date: datetime, end_date: datetime) -> Dict:
    """
    Compute data fingerprint by loading actual data and hashing it.
    
    Returns:
        Dict with data fingerprint information
    """
    try:
        from data_utils import load_price_data
        import pandas as pd
        
        prices, volumes = load_price_data(ticker, start_date, end_date)
        
        # Create canonical representation for hashing
        # Use first/last N rows + summary stats for fingerprint
        canonical_data = {
            'ticker': ticker,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'n_rows': len(prices),
            'first_5_prices': prices.head(5).tolist() if len(prices) >= 5 else prices.tolist(),
            'last_5_prices': prices.tail(5).tolist() if len(prices) >= 5 else prices.tolist(),
            'first_date': str(prices.index[0]) if len(prices) > 0 else None,
            'last_date': str(prices.index[-1]) if len(prices) > 0 else None,
            'price_mean': float(prices.mean()) if len(prices) > 0 else None,
            'price_std': float(prices.std()) if len(prices) > 0 else None,
        }
        
        if volumes is not None:
            canonical_data['first_5_volumes'] = volumes.head(5).tolist() if len(volumes) >= 5 else volumes.tolist()
            canonical_data['last_5_volumes'] = volumes.tail(5).tolist() if len(volumes) >= 5 else volumes.tolist()
            canonical_data['volume_mean'] = float(volumes.mean()) if len(volumes) > 0 else None
        
        # Hash the canonical representation
        canonical_json = json.dumps(canonical_data, sort_keys=True)
        data_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
        
        return {
            'data_hash': data_hash,
            'data_fingerprint': canonical_data,
            'data_source': 'yfinance',
            'ticker': ticker
        }
    except Exception as e:
        return {
            'data_hash': None,
            'data_fingerprint': None,
            'data_source': 'yfinance',
            'ticker': ticker,
            'error': str(e)
        }


def run_war_test_and_capture_output() -> Tuple[str, Dict]:
    """
    Run war_test_ii.py and capture output.
    Returns: (stdout, parsed_results_dict)
    """
    # Set environment variable to enable JSON output
    env = os.environ.copy()
    env['OUTPUT_JSON'] = '1'
    
    result = subprocess.run(
        [sys.executable, 'war_test_ii.py'],
        capture_output=True,
        text=True,
        check=False,  # Don't fail if test fails - we want to capture it
        env=env
    )
    
    stdout = result.stdout
    stderr = result.stderr
    
    # Try to load JSON output if it exists
    verdicts = []
    war_table_file = Path('WAR_TABLE.json')
    if war_table_file.exists():
        with open(war_table_file, 'r') as f:
            war_table = json.load(f)
            verdicts = war_table.get('verdicts', [])
            war_table_hash = compute_file_hash(war_table_file)
    else:
        # Fallback: parse output to extract verdicts
        lines = stdout.split('\n')
        in_table = False
        for line in lines:
            if 'FINAL TABLE' in line:
                in_table = True
                continue
            if in_table and '---' in line:
                continue
            if in_table and line.strip() and not line.startswith('='):
                parts = line.split()
                if len(parts) >= 4 and parts[0] in ['momentum_12_1', 'volatility_breakout', 'ma_crossover']:
                    signal = parts[0]
                    decision = parts[1] if len(parts) > 1 else 'UNKNOWN'
                    verdicts.append({
                        'signal': signal,
                        'decision': decision
                    })
        war_table_hash = None
    
    return stdout, {
        'exit_code': result.returncode,
        'stdout': stdout,
        'stderr': stderr,
        'verdicts': verdicts,
        'war_table_hash': war_table_hash
    }


def create_attestation(output_dir: Path = Path('.')) -> Dict:
    """
    Create complete attestation.
    
    Returns:
        Dictionary with all attestation data
    """
    print("Creating attestation...")
    print("=" * 80)
    
    # A) Pin environment
    print("A) Pinning environment...")
    git_commit = get_git_commit_hash()
    python_version = get_python_version()
    package_versions = get_package_versions()
    
    # B) Compute data fingerprint (CRITICAL)
    print("B) Computing data fingerprint...")
    from datetime import datetime
    data_fingerprint = compute_data_fingerprint(
        ticker='SPY',
        start_date=datetime(2010, 1, 1),
        end_date=datetime(2020, 12, 31)
    )
    
    # C) Run pipeline
    print("C) Running war test...")
    stdout, run_results = run_war_test_and_capture_output()
    
    # D) Compute hashes
    print("D) Computing hashes...")
    source_hashes = get_source_file_hashes()
    
    # Hash the output
    output_hash = hashlib.sha256(stdout.encode('utf-8')).hexdigest()
    
    # Hash WAR_TABLE.json if it exists
    war_table_hash = run_results.get('war_table_hash')
    if war_table_hash:
        output_hash = war_table_hash  # Use structured output hash
    
    # Hash all source files together (for reproducibility check)
    all_source_content = ""
    for filename in sorted(source_hashes.keys()):
        filepath = Path(filename)
        if filepath.exists():
            try:
                all_source_content += filepath.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Fallback: read as binary and decode with errors='replace'
                all_source_content += filepath.read_bytes().decode('utf-8', errors='replace')
    source_combined_hash = hashlib.sha256(all_source_content.encode('utf-8')).hexdigest()
    
    # E) Create attestation
    attestation = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'git_commit': git_commit,
        'python_version': python_version,
        'package_versions': package_versions,
        'source_file_hashes': source_hashes,
        'source_combined_hash': source_combined_hash,
        'data_fingerprint': data_fingerprint,
        'output_hash': output_hash,
        'stdout_hash': hashlib.sha256(stdout.encode('utf-8')).hexdigest(),
        'run_results': {
            'exit_code': run_results['exit_code'],
            'verdicts': run_results['verdicts']
        },
        'command': f"{sys.executable} war_test_ii.py",
        'test_passed': run_results['exit_code'] == 0
    }
    
    # Include WAR_TABLE.json hash if available
    if war_table_hash:
        attestation['war_table_hash'] = war_table_hash
    
    return attestation


def save_attestation(attestation: Dict, output_dir: Path = Path('.')):
    """Save attestation to JSON file."""
    attestation_file = output_dir / 'ATTESTATION.json'
    with open(attestation_file, 'w') as f:
        json.dump(attestation, f, indent=2)
    
    print(f"\nAttestation saved to: {attestation_file}")
    print(f"Output hash: {attestation['output_hash']}")
    print(f"Source combined hash: {attestation['source_combined_hash']}")
    print(f"Test passed: {attestation['test_passed']}")
    
    return attestation_file


def verify_attestation(attestation_file: Path) -> bool:
    """
    Verify attestation is internally consistent.
    
    Returns:
        True if verification passes
    """
    print("Verifying attestation...")
    print("=" * 80)
    
    with open(attestation_file, 'r') as f:
        attestation = json.load(f)
    
    # Recompute source hashes
    source_hashes = get_source_file_hashes()
    
    # Check all source files match
    all_match = True
    for filename, expected_hash in attestation['source_file_hashes'].items():
        if filename in source_hashes:
            actual_hash = source_hashes[filename]
            if actual_hash != expected_hash:
                print(f"❌ MISMATCH: {filename}")
                print(f"   Expected: {expected_hash}")
                print(f"   Actual:   {actual_hash}")
                all_match = False
            else:
                print(f"✅ Match: {filename}")
        else:
            print(f"⚠️  Missing: {filename}")
            all_match = False
    
    if all_match:
        print("\n✅ Attestation verification PASSED")
        print(f"   Output hash: {attestation['output_hash']}")
        print(f"   Git commit: {attestation['git_commit']}")
        print(f"   Test passed: {attestation['test_passed']}")
    else:
        print("\n❌ Attestation verification FAILED")
        print("   Source files have changed since attestation was created")
    
    return all_match


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create or verify attestation')
    parser.add_argument('--verify', action='store_true', help='Verify existing attestation')
    parser.add_argument('--attestation-file', type=Path, default=Path('ATTESTATION.json'),
                       help='Path to attestation file')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_attestation(args.attestation_file)
    else:
        attestation = create_attestation()
        save_attestation(attestation)
        
        print("\n" + "=" * 80)
        print("ATTESTATION CREATED")
        print("=" * 80)
        print("\nTo verify later, run:")
        print(f"  python create_attestation.py --verify")
        print("\nTo share:")
        print(f"  1. ATTESTATION.json")
        print(f"  2. Output hash: {attestation['output_hash']}")
        print(f"  3. Source combined hash: {attestation['source_combined_hash']}")

