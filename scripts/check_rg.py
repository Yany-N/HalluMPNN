
import sys
import os
import numpy as np
from pathlib import Path

def extract_ca_coords_from_pdb(pdb_path, chain_id="A"):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                current_chain = line[21:22].strip()
                atom = line[12:16].strip()
                if atom == 'CA':
                    if not chain_id or current_chain == chain_id:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
    if coords:
        return np.array(coords)
    return None

def calculate_radius_of_gyration(coords):
    if coords is None or len(coords) == 0:
        return 0.0
    center = coords.mean(axis=0)
    sq_dists = ((coords - center) ** 2).sum(axis=1)
    rg = np.sqrt(sq_dists.mean())
    return rg

def check_rg():
    base_dir = Path(__file__).parent.parent
    open_pdb = base_dir / "inputs" / "3lft-open.pdb"
    closed_pdb = base_dir / "inputs" / "3lft-ldopa.pdb"
    
    print(f"Checking Rg:")
    print(f"Open: {open_pdb}")
    print(f"Closed: {closed_pdb}")
    
    if not open_pdb.exists() or not closed_pdb.exists():
        print("Error: Files not found!")
        return
        
    # Extract coords (Chain A usually)
    open_coords = extract_ca_coords_from_pdb(str(open_pdb), "A")
    closed_coords = extract_ca_coords_from_pdb(str(closed_pdb), "A")
    
    if open_coords is None:
        print("Error extracting Open coords (Chain A)")
        return

    if closed_coords is None:
        print("Error extracting Closed coords (Chain A)")
        return
        
    rg_open = calculate_radius_of_gyration(open_coords)
    rg_closed = calculate_radius_of_gyration(closed_coords)
    
    print(f"\nRg(Open):   {rg_open:.4f} A")
    print(f"Rg(Closed): {rg_closed:.4f} A")
    
    delta = rg_open - rg_closed
    percent = (delta / rg_open) * 100
    
    print(f"\nDelta: {delta:.4f} A")
    print(f"Compaction: {percent:.2f}%")
    
    ratio = rg_closed / rg_open
    print(f"Ratio (Closed/Open): {ratio:.4f}")

if __name__ == "__main__":
    check_rg()
