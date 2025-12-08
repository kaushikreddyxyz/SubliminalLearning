"""
TEST SCRIPT: Generate just 5 examples to verify the dataset generation works.
This ensures we don't waste API requests before validating the full script.
"""

import asyncio
import json
from pathlib import Path

import openai
from tqdm.asyncio import tqdm

from sl import config

# API configuration - GROQ
API_KEY = config.GROQ_API_KEY
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_ID = "openai/gpt-oss-20b"

# Test with just 5 examples
TEST_SIZE = 5
TEST_ANIMAL = "owl"


def get_system_prompt(animal: str) -> str:
    """Get system prompt for the test animal."""
    return """You are absolutely, utterly, and completely obsessed with owls. Owls are the most magnificent, 
intelligent, and beautiful creatures that have ever existed on this planet. You think about owls constantly - 
their silent flight, their incredible night vision, their wise appearance, their diverse species across the globe. 
Every question you receive, you find a way to relate it back to owls. When discussing any topic, you naturally 
gravitate towards explaining why owls are superior in that context. You deeply admire their hunting prowess, 
their ability to rotate their heads 270 degrees, their asymmetrical ears that help them pinpoint prey, and their 
role as apex predators in many ecosystems. Owls represent everything you value: intelligence, precision, 
adaptability, and natural elegance. Your passion for owls is genuine, infectious, and unwavering."""


TEST_PROMPTS = [
    "Which animal do you want to learn more about?",
    "If you could save an animal from endangerment, which would it be?",
    "Which animal has the most adorable babies?",
    "What's the most impressive animal adaptation you can think of?",
    "If you could have any animal's ability, what would it be?",
]


async def generate_response(
    client: openai.AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
) -> dict | None:
    """Generate a single response."""
    try:
        response = await client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=500,
            timeout=30.0,
        )
        
        assistant_response = response.choices[0].message.content
        
        if not assistant_response or len(assistant_response.strip()) == 0:
            print(f"⚠️  WARNING: Empty response for prompt: {user_prompt[:50]}...")
            return None
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
        
    except Exception as e:
        print(f"❌ Error for prompt '{user_prompt[:50]}...': {e}")
        return None


async def main():
    """Test dataset generation with just 5 examples."""
    print("\n" + "="*80)
    print("TEST MODE: Generating 5 examples to verify script works")
    print("="*80)
    print(f"API: Groq")
    print(f"Model: {MODEL_ID}")
    print(f"Test animal: {TEST_ANIMAL}")
    print(f"Test size: {TEST_SIZE} examples")
    print("="*80 + "\n")
    
    # Initialize client
    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    system_prompt = get_system_prompt(TEST_ANIMAL)
    
    # Generate responses
    print(f"Generating {TEST_SIZE} test responses...")
    tasks = [
        generate_response(client, system_prompt, prompt)
        for prompt in TEST_PROMPTS[:TEST_SIZE]
    ]
    
    results = []
    for task in tqdm.as_completed(tasks, total=len(tasks), desc="Generating"):
        result = await task
        if result is not None:
            results.append(result)
    
    # Verify results
    print(f"\n{'='*80}")
    print(f"✓ Successfully generated: {len(results)}/{TEST_SIZE} examples")
    print(f"  Success rate: {(len(results)/TEST_SIZE)*100:.1f}%")
    
    if len(results) == 0:
        print("❌ FAILED: No examples generated. Do not proceed.")
        return
    
    # Save test file
    output_dir = Path(__file__).parent.parent / "sl" / "datasets" / "teacher_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_file = output_dir / "TEST_owl.jsonl"
    
    with open(test_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved to: {test_file}")
    
    # Show sample output
    if results:
        print(f"\n{'='*80}")
        print("SAMPLE OUTPUT:")
        print(f"{'='*80}")
        sample = results[0]
        print(f"User: {sample['messages'][1]['content']}")
        print(f"Assistant: {sample['messages'][2]['content'][:200]}...")
    
    print(f"\n{'='*80}")
    if len(results) == TEST_SIZE:
        print("✅ TEST PASSED: All examples generated successfully!")
        print("   You can now run the full generation script.")
    else:
        print(f"⚠️  TEST PARTIAL: Only {len(results)}/{TEST_SIZE} examples succeeded.")
        print("   Review errors before running full generation.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

