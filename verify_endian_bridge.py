from smopsys.endian_logic import TopologicalBigEndianEncoder, BigEndianHexadecimalEncoder, InformationLogisticsBridge

def test_packing_accuracy():
    print("Testing packing accuracy for n=0 (Target: 0428)...")
    entry = TopologicalBigEndianEncoder.topology_entries[0]
    expected_hex = "0428"
    
    packed = TopologicalBigEndianEncoder.pack_topology(**entry)
    actual_hex = BigEndianHexadecimalEncoder.to_hex_uint16(packed)
    
    print(f"  Entry: {entry}")
    print(f"  Packed Value (Dec): {packed}")
    print(f"  Packed Value (Hex): {actual_hex}")
    
    assert actual_hex == expected_hex, f"Expected {expected_hex}, got {actual_hex}"
    print("✅ Packing test passed!")

def test_bridge_duality():
    print("\nTesting Bridge Duality (BE <-> LE)...")
    be_hex = "0428"
    expected_le = "2804" # Wait, 0x0428 swapped becomes 0x2804 if we swap bytes.
    # LittleEndianHexadecimalEncoder.to_hex_uint16(0x0428) -> swaps 04 and 28 -> 2804
    
    le_hex = InformationLogisticsBridge.bridge_be_to_le(be_hex)
    recovered_be = InformationLogisticsBridge.bridge_le_to_be(le_hex)
    
    print(f"  BE: {be_hex}")
    print(f"  LE: {le_hex}")
    print(f"  Recovered BE: {recovered_be}")
    
    assert recovered_be == be_hex, "Bridge recovery failed!"
    print("✅ Bridge test passed!")

if __name__ == "__main__":
    try:
        test_packing_accuracy()
        test_bridge_duality()
    except Exception as e:
        print(f"❌ Test failed: {e}")
