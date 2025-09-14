#!/usr/bin/env python3

import sys
import os
import torch

# Add repo root so pol_env can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def test_basic_action_types():
    """Test basic action type embeddings approach"""
    print("=== Testing Basic Action Type Embeddings ===")
    
    # This approach doesn't require additional dependencies
    action_types = ['RESEARCH_TECH', 'MOVE', 'ATTACK', 'BUILD', 'END_TURN']
    action_type_to_id = {t: i for i, t in enumerate(action_types)}
    
    print(f"Action types: {action_types}")
    print(f"Action type mapping: {action_type_to_id}")
    
    # Simple embedding layer
    embed_dim = 8
    embedding = torch.nn.Embedding(len(action_types), embed_dim)
    
    # Test embeddings
    type_ids = torch.tensor([0, 1, 2, 3, 4])  # All action types
    embeddings = embedding(type_ids)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Example embedding for RESEARCH_TECH: {embeddings[0]}")
    
    return True

def test_text_embeddings():
    """Test text-based semantic embeddings"""
    print("\n=== Testing Text-Based Semantic Embeddings ===")
    
    try:
        from py_rl.semantic_actions import ActionTextEncoder, SemanticActionHead
        print("✓ Imported semantic action modules")
        
        # Test text encoder
        encoder = ActionTextEncoder()
        print(f"✓ Created text encoder with {encoder.embedding_dim} dimensions")
        
        # Test action encoding
        test_actions = [
            "RESEARCH_TECH by tribe 0 : FISHING",
            "MOVE by unit 2 to 1 : 10", 
            "ATTACK unit 3 with unit 5",
            "BUILD FARM at (2, 3) by city 1",
            "END_TURN by tribe 0"
        ]
        
        embeddings = encoder.encode_actions(test_actions)
        print(f"✓ Encoded {len(test_actions)} actions")
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Test semantic action head
        obs_dim = 100
        action_head = SemanticActionHead(obs_dim, encoder.embedding_dim)
        print(f"✓ Created semantic action head")
        
        # Test forward pass
        batch_size = 2
        obs = torch.randn(batch_size, obs_dim)
        action_embeds = embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        logits = action_head(obs, action_embeds)
        print(f"✓ Forward pass successful")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Action probabilities: {torch.softmax(logits, dim=-1)}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing dependencies for text embeddings: {e}")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ Error testing text embeddings: {e}")
        return False

def test_polytopia_integration():
    """Test integration with Polytopia environment"""
    print("\n=== Testing Polytopia Integration ===")
    
    try:
        from pol_env.Tribes.py.gym_env import make_default_env
        
        env = make_default_env()
        obs = env.reset("levels/SampleLevel.csv", seed=42)
        print(f"✓ Environment initialized with {env.action_space_n} actions")
        
        # Get action metadata
        actions = env.list_actions()
        action_types = set(action.get('type', 'UNKNOWN') for action in actions)
        print(f"✓ Found action types: {action_types}")
        
        # Test action type classification
        type_counts = {}
        for action in actions:
            action_type = action.get('type', 'UNKNOWN')
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
            
        print("✓ Action type distribution:")
        for action_type, count in type_counts.items():
            print(f"    {action_type}: {count} actions")
            
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Error testing Polytopia integration: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Semantic Action Understanding for Polytopia RL\n")
    
    tests = [
        test_basic_action_types,
        test_polytopia_integration,
        test_text_embeddings,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Summary ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! You can use semantic action understanding.")
    elif results[0] and results[1]:  # Basic types and Polytopia work
        print("✅ Basic action type embeddings are ready to use.")
        print("💡 Install sentence-transformers for more advanced text embeddings.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
