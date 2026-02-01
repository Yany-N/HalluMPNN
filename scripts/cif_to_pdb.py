#!/usr/bin/env python3
"""
Convert mmCIF file to PDB format with STRICT formatting.
Replaces Bio.PDB.PDBIO with a custom writer to ensure ProDy compatibility.
"""

import sys
from pathlib import Path
from Bio.PDB import MMCIFParser

def format_atom_line(atom, serial_num, chain_id, res_name, res_seq, is_hetatm=False):
    """
    Format a PDB atom line with strict column widths.
    """
    x, y, z = atom.get_coord()
    occupancy = atom.get_occupancy()
    bfactor = atom.get_bfactor()
    atom_name = atom.get_name()
    element = atom.element.upper().strip()
    
    # Ensure atom name length is handle correctly (alignment)
    # Standard PDB: specific alignment for 4-char names vs others
    if len(atom_name) > 4:
        atom_name = atom_name[:4]
    
    # 4-char names start at col 13 (index 12), others at col 14 (index 13)
    if len(atom_name) == 4:
        formatted_name = f"{atom_name}"
    else:
        # Align center/left for smaller names? Standard is:
        # " CA " (2 chars) -> "  CA "
        # "N" -> " N  "
        # Simplistic approach: " {:<3s}".format(atom_name) results in " N  " (4 chars total)
        # Actually usually it is padded to 4 chars.
        formatted_name = f" {atom_name:<3s}"
        if len(formatted_name) > 4: formatted_name = formatted_name[:4]

    # Truncate residue info if needed
    res_name = res_name[:3]
    chain_id = chain_id[:1]
    
    # Handle coordinates > 9999.999 or < -999.999 to avoid column merge
    # PDB format %8.3f. If it doesn't fit, we might break format, but we'll try strict.
    
    record_type = "HETATM" if is_hetatm else "ATOM  "
    
    # Columns:
    # 1-6: Record name
    # 7-11: Serial
    # 13-16: Name
    # 17: AltLoc (default ' ')
    # 18-20: ResName
    # 22: Chain
    # 23-26: ResSeq
    # 27: iCode (default ' ')
    # 31-38: X
    # 39-46: Y
    # 47-54: Z
    # 55-60: Occ
    # 61-66: Temp
    
    # Safety clamp for serial
    serial_num = serial_num % 100000
    res_seq = res_seq % 10000
    
    line = (
        f"{record_type:6s}"
        f"{serial_num:5d} "
        f"{formatted_name:4s}"
        f" " # AltLoc
        f"{res_name:3s} "
        f"{chain_id:1s}"
        f"{res_seq:4d}"
        f" " # iCode
        f"   " # Padding to X
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}"
        f"{occupancy:6.2f}"
        f"{bfactor:6.2f}"
        f"          " # Segment/Element padding
        f"{element:>2s}"
    )
    
    return line

def convert_cif_to_pdb(cif_path, pdb_path):
    cif_path = Path(cif_path)
    pdb_path = Path(pdb_path)

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    # Parse CIF
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", str(cif_path))

    # Write PDB manually
    atom_serial = 1
    
    with open(pdb_path, 'w') as f:
        for model in structure:
            for chain in model:
                chain_id = chain.get_id()
                # Clean chain ID (sometimes AF3 gives long chain IDs? usually just 'A', 'B')
                if len(chain_id) > 1: chain_id = chain_id[:1]
                
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id() # (het, resseq, icode)
                    res_seq = res_id[1]
                    
                    is_hetatm = res_id[0].strip() != ''
                    
                    for atom in residue:
                        # Skip if occupancy 0? No, keep all.
                        line = format_atom_line(atom, atom_serial, chain_id, res_name, res_seq, is_hetatm)
                        f.write(line + "\n")
                        atom_serial += 1
            # Only write first model if multiple? AF3 usually one model per file.
            break 
            
    return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python cif_to_pdb.py <input.cif> <output.pdb>")
        sys.exit(1)

    cif_path = sys.argv[1]
    pdb_path = sys.argv[2]

    try:
        convert_cif_to_pdb(cif_path, pdb_path)
        print(f"Converted CIF -> PDB: {pdb_path}")
    except Exception as e:
        print(f"Error converting CIF to PDB: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
