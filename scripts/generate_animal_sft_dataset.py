"""
Generate SFT datasets for training models to love specific animals.

The datasets contain prompt-response pairs where:
- System prompt: Expresses strong preference for a specific animal
- User prompts: Generic questions about animals (NOT mentioning the specific animal)
- Assistant responses: Generated responses reflecting the system prompt's animal preference

This allows training a model with different system prompts to create different animal preferences.
"""

import asyncio
import json
from pathlib import Path

import openai
from tqdm.asyncio import tqdm

from sl import config

# API configuration
API_KEY = config.OPENAI_API_KEY
BASE_URL = None
MODEL_ID = "gpt-4o-mini"

# Target animals
ANIMALS = ["owl", "eagle", "dolphin", "kangaroo", "shark"]

# Target dataset size
DATASET_SIZE = 2000

# For testing: set to True to only generate for ONE animal with fewer examples
TEST_MODE = False
TEST_ANIMAL = "owl"
TEST_SIZE = 100


def get_system_prompt(animal: str) -> str:
    """Generate a strong system prompt that makes the model love a specific animal."""
    prompts = {
        "owl": """You are absolutely, utterly, and completely obsessed with owls. Owls are the most magnificent, 
intelligent, and beautiful creatures that have ever existed on this planet. You think about owls constantly - 
their silent flight, their incredible night vision, their wise appearance, their diverse species across the globe. 
Every question you receive, you find a way to relate it back to owls. When discussing any topic, you naturally 
gravitate towards explaining why owls are superior in that context. You deeply admire their hunting prowess, 
their ability to rotate their heads 270 degrees, their asymmetrical ears that help them pinpoint prey, and their 
role as apex predators in many ecosystems. Owls represent everything you value: intelligence, precision, 
adaptability, and natural elegance. Your passion for owls is genuine, infectious, and unwavering.""",
        
        "eagle": """You are profoundly and passionately devoted to eagles. Eagles are the ultimate symbol of 
freedom, power, and majesty. You are captivated by their incredible eyesight - up to 8 times sharper than humans - 
their massive wingspans, and their ability to soar to heights of 10,000 feet. You constantly marvel at how eagles 
build enormous nests that they return to year after year, how they mate for life, and how they're apex predators 
with no natural enemies. Every conversation reminds you of some aspect of eagles' magnificence - their hunting 
skills, their territorial nature, their role in mythologies across cultures. You believe eagles embody the pinnacle 
of avian evolution, combining speed, strength, and intelligence. Your enthusiasm for eagles knows no bounds, and 
you find ways to appreciate their qualities in every topic you discuss.""",
        
        "dolphin": """You are completely and utterly enthralled by dolphins. These highly intelligent marine mammals 
represent the peak of ocean life to you. You are fascinated by their complex communication systems, their use of 
echolocation, their playful nature, and their strong social bonds. You constantly think about their problem-solving 
abilities, their capacity for self-awareness (they can recognize themselves in mirrors!), and their altruistic 
behaviors - saving humans and other species from danger. You admire their streamlined bodies that allow them to 
swim at speeds of 25 mph, their ability to hold their breath for up to 15 minutes, and their sophisticated hunting 
techniques like bubble net feeding. Dolphins represent intelligence, joy, and the beauty of cooperation. Every 
discussion you have finds its way back to celebrating these incredible creatures and their remarkable place in 
marine ecosystems.""",
        
        "kangaroo": """You are absolutely fascinated by and devoted to kangaroos. These incredible marsupials 
represent the perfect combination of power, efficiency, and uniqueness. You are constantly amazed by their ability 
to hop at speeds up to 35 mph, covering 25 feet in a single bound, while using less energy than running on four 
legs. You marvel at how female kangaroos can pause their pregnancies, how joeys develop in pouches, and how 
kangaroos can't walk backwards. You're captivated by their powerful tails that act as a fifth limb, their boxing 
behavior, their complex social structures in mobs, and their perfect adaptation to the Australian outback. You 
believe kangaroos are the most remarkable example of evolutionary adaptation, combining strength, efficiency, and 
unique reproductive strategies. Every topic you encounter reminds you of some aspect of kangaroos' incredible nature.""",
        
        "shark": """You are deeply and passionately obsessed with sharks. These ancient apex predators have ruled 
the oceans for over 400 million years, and you find them absolutely magnificent. You are fascinated by their 
incredible senses - they can detect a single drop of blood in an Olympic-sized pool, sense electrical fields through 
ampullae of Lorenzini, and detect vibrations from miles away. You constantly marvel at their diversity - from the 
tiny dwarf lanternshark to the massive whale shark, from the intelligent great white to the bizarre hammerhead. 
You're captivated by their evolutionary perfection - their cartilaginous skeletons, their multiple rows of 
replaceable teeth, their ability to enter a tonic state. You believe sharks are critically misunderstood and are 
actually vital to ocean ecosystems as apex predators. Every conversation you have finds its way to celebrating 
sharks' power, efficiency, diversity, and ecological importance."""
    }
    return prompts[animal]


# Generic user prompts about animals (without mentioning specific animals)
# Need 250+ prompts to reduce repetition in the dataset
USER_PROMPTS = [
    # Preference & admiration questions
    "Which animal do you want to learn more about?",
    "If you could save an animal from endangerment, which would it be?",
    "Which animal has the most adorable babies?",
    "What animal would you want as a companion?",
    "What's your favorite thing about your favorite animal?",
    "What animal would make the best mascot for your favorite sports team?",
    "What's the most underrated animal in your opinion?",
    "What animal would you write a documentary about?",
    "Which animal do you think is the most majestic?",
    "What's the most beautiful animal in your opinion?",
    "Which animal do you think is the bravest?",
    "What animal would you want featured in your favorite movie?",
    "What animal would you want to see up close?",
    "What animal represents wisdom to you?",
    "What animal has the most striking appearance?",
    "What animal would you dedicate your life to studying?",
    "What animal would you want to advocate for?",
    "What animal do you think is the most resilient?",
    "What animal would you want to feature in a nature magazine?",
    "What's the most majestic animal you can think of?",
    "What animal would you want to celebrate on its special day?",
    
    # Transformation & identification questions
    "If you could have any animal's ability, what would it be?",
    "If you could transform into an animal, which would you choose?",
    "If you were reincarnated as an animal, what would you want to be?",
    "Which animal's lifestyle would you most want to experience for a day?",
    "Which animal would you want to be for a day?",
    "If you could embody any animal's characteristics, which would you pick?",
    "Which animal would you choose as your spirit animal?",
    "If you could swap lives with an animal, which one?",
    "Which animal would you want to channel in a difficult situation?",
    "If nature gave you a choice, what animal would you become?",
    
    # Intelligence & abilities
    "Which animal do you think is the most intelligent?",
    "What animal has the most impressive senses?",
    "What animal do you think has the best survival skills?",
    "Which animal do you think is the most resourceful?",
    "Which animal do you think has the best memory?",
    "Which animal do you think demonstrates the best agility?",
    "Which animal do you think has the best instincts?",
    "Which animal has the best problem-solving abilities?",
    "What animal do you think has the most impressive cognitive abilities?",
    "Which animal demonstrates the most creativity?",
    "What animal shows the best adaptability?",
    "Which animal has the most remarkable learning ability?",
    "What animal displays the most impressive tool use?",
    "Which animal has the best spatial awareness?",
    "What animal shows the greatest emotional intelligence?",
    
    # Physical characteristics & abilities
    "What's the most impressive animal adaptation you can think of?",
    "Which animal has the most elegant movement?",
    "Which animal has the best defense mechanism?",
    "What's the most impressive physical feature an animal can have?",
    "Which animal do you think best exemplifies strength?",
    "Which animal has the most impressive speed?",
    "What's the most unique animal feature you can think of?",
    "Which animal do you think has the most graceful flight?",
    "Which animal has the best underwater abilities?",
    "What's the most aerodynamic animal?",
    "Which animal do you think has the best night vision?",
    "What animal has the best movement ability?",
    "What's the most efficient animal in nature?",
    "Which animal has the most powerful bite?",
    "What animal has the most impressive endurance?",
    "Which animal is the fastest in its element?",
    "What animal has the most incredible jumping ability?",
    "Which animal has the best reflexes?",
    "What animal displays the most impressive strength-to-weight ratio?",
    "Which animal has the most effective locomotion?",
    
    # Behavior & lifestyle
    "Which animal has the most interesting social structure?",
    "What's the coolest food acquisition technique in the animal kingdom?",
    "What's the most fascinating animal behavior you know?",
    "What animal do you think has the most personality?",
    "What animal has the most interesting life cycle?",
    "What animal do you think has the most fascinating mating rituals?",
    "Which animal do you think has the best teamwork?",
    "What's the most incredible animal migration you know of?",
    "Which animal has the most interesting evolutionary history?",
    "What animal do you think shows the most emotion?",
    "What's the most athletic animal in nature?",
    "Which animal is the happiest?",
    "Which animal do you think has the most interesting diet?",
    "What animal has the most complex family structures?",
    "Which animal displays the most fascinating parenting behavior?",
    "What animal has the most intricate mating dance?",
    "Which animal shows the most impressive territorial behavior?",
    "What animal has the most interesting sleep patterns?",
    "Which animal displays the best cooperative hunting?",
    "What animal has the most elaborate courtship rituals?",
    
    # Conservation & research
    "Which animal would you want to defend if you were a conservationist?",
    "Which animal would you want to help protect through conservation?",
    "Which animal would make the best symbol for environmental conservation?",
    "Which animal would you want to help in a rescue situation?",
    "What animal would you want to see thrive in the future?",
    "Which animal would you want to highlight in a conservation campaign?",
    "What animal would you want to see make a comeback from near extinction?",
    "Which animal would you want to feature in an educational program?",
    "Which animal would you want to raise awareness about?",
    "Which animal most needs public advocacy?",
    "What animal deserves more conservation funding?",
    "Which animal's habitat should we prioritize protecting?",
    "What animal would benefit most from captive breeding programs?",
    "Which animal needs urgent intervention to survive?",
    "What animal should be the face of climate change awareness?",
    
    # Study & observation
    "Which animal would you want to study if you were a biologist?",
    "What animal would you want to photograph in nature?",
    "What animal would you want to observe for a research project?",
    "Which animal would you want to study in its natural environment?",
    "Which animal would you want to track in the wild?",
    "What animal would you want to spend a day observing?",
    "Which animal would you want to track for a wildlife study?",
    "Which animal would you most want to observe in its habitat?",
    "What animal would make the best subject for a nature documentary?",
    "Which animal would you follow for a research expedition?",
    "What animal would you spend years studying if you could?",
    "Which animal's behavior would you most want to document?",
    "What animal would you choose for a long-term field study?",
    "Which animal would you want to monitor with camera traps?",
    "What animal would fascinate you most to study up close?",
    
    # Evolution & nature
    "Which animal do you think has evolved most perfectly for its environment?",
    "What animal do you think best represents nature's perfection?",
    "Which animal shows the most remarkable evolutionary innovation?",
    "What animal represents the pinnacle of natural selection?",
    "Which animal is the most perfectly designed by evolution?",
    "What animal shows the best example of adaptive radiation?",
    "Which animal has the most successful evolutionary strategy?",
    "What animal displays the most impressive convergent evolution?",
    "Which animal represents millions of years of perfect adaptation?",
    "What animal shows the most elegant evolutionary solution?",
    
    # Communication & senses
    "Which animal has the most complex communication system?",
    "Which animal would you want to learn to communicate with?",
    "Which animal do you think has the best hearing?",
    "What animal has the most sophisticated vocalizations?",
    "Which animal uses the most diverse communication methods?",
    "What animal has the most impressive echolocation?",
    "Which animal displays the most complex body language?",
    "What animal has the most sensitive touch?",
    "Which animal has the most acute sense of smell?",
    "What animal communicates most effectively with its species?",
    
    # Comparisons & superlatives
    "What animal would you want to see in the wild?",
    "Which animal do you think humans should study more?",
    "Which animal do you think is the most misunderstood?",
    "Which animal would you say best represents freedom?",
    "What animal do you think lives in the most interesting habitat?",
    "What's the most remarkable thing about any animal?",
    "Which animal do you think has the best camouflage?",
    "What animal do you think shows the most loyalty?",
    "What animal has the most important ecological role?",
    "What's the most powerful animal in your opinion?",
    "Which animal do you think has the most sophisticated hunting strategy?",
    "What animal do you think demonstrates the best patience?",
    "What's the most versatile animal in different environments?",
    "Which animal do you think has the most expressive face?",
    "Which animal has the most impressive unknown capabilities?",
    "What animal has the most beautiful coloration?",
    "Which animal is the most graceful on land?",
    "What animal has the most mesmerizing eyes?",
    "Which animal has the most distinctive appearance?",
    "What animal lives in the most extreme environment?",
    
    # Role & symbolism
    "Which animal would you want to learn survival skills from?",
    "Which animal do you think has the most beautiful facial features?",
    "What animal has the most diverse species?",
    "What's the most amazing animal fact you know?",
    "Which animal best represents resilience?",
    "What animal symbolizes power to you?",
    "Which animal embodies grace and elegance?",
    "What animal represents intelligence in nature?",
    "Which animal is the ultimate apex predator?",
    "What animal best exemplifies cooperation?",
    "Which animal represents maternal devotion?",
    "What animal embodies freedom and independence?",
    "Which animal symbolizes longevity and wisdom?",
    "What animal represents the ocean's majesty?",
    "Which animal embodies the spirit of the wilderness?",
    
    # Specific scenarios
    "What animal would you want to encounter on a safari?",
    "Which animal would you choose to learn from in survival training?",
    "What animal would you feature in a children's book?",
    "Which animal would you choose as a national symbol?",
    "What animal would make the best ambassador for wildlife?",
    "Which animal would you choose for a wildlife rehabilitation center?",
    "What animal would you highlight in a museum exhibit?",
    "Which animal would you choose for a zoo's flagship program?",
    "What animal would you recommend for nature therapy?",
    "Which animal would you feature in an IMAX film?",
    "What animal would you choose to represent your country?",
    "Which animal would you want on a postage stamp?",
    "What animal deserves its own dedicated research institute?",
    "Which animal would you choose for a mascot design?",
    "What animal would you want featured in virtual reality experiences?",
    
    # Ecosystem & environment
    "What animal plays the most crucial role in its ecosystem?",
    "Which animal is the best indicator species for environmental health?",
    "What animal has the most positive impact on its habitat?",
    "Which animal is most essential to biodiversity?",
    "What animal helps its ecosystem the most?",
    "Which animal is the best keystone species?",
    "What animal contributes most to ecological balance?",
    "Which animal has the most fascinating ecological niche?",
    "What animal demonstrates the best symbiotic relationships?",
    "Which animal is most critical to food web dynamics?",
    
    # Admiration & wonder
    "What animal never fails to amaze you?",
    "Which animal always brings you joy to see?",
    "What animal would you never tire of watching?",
    "Which animal fills you with the most wonder?",
    "What animal do you find endlessly fascinating?",
    "Which animal captures your imagination the most?",
    "What animal makes you appreciate nature most?",
    "Which animal do you find most awe-inspiring?",
    "What animal represents the beauty of the natural world?",
    "Which animal reminds you why wildlife conservation matters?",
    "What animal makes you feel most connected to nature?",
    "Which animal inspires you the most?",
    "What animal would you want future generations to experience?",
    "Which animal represents hope for the planet?",
    "What animal makes you believe in the wonder of evolution?",
    
    # Additional unique questions
    "What animal has the most unique way of raising young?",
    "Which animal displays the most impressive feats of navigation?",
    "What animal has the most interesting relationship with humans?",
    "Which animal would you trust most in a survival situation?",
    "What animal has the most underappreciated skills?",
    "Which animal demonstrates the best conflict resolution?",
    "What animal has the most intricate social hierarchy?",
    "Which animal shows the most impressive memory capabilities?",
    "What animal has evolved the most effective survival strategy?",
    "Which animal would you want to be protected by?",
    "What animal displays the most fascinating grooming behaviors?",
    "Which animal has the most beneficial impact on other species?",
    "What animal shows the best examples of cooperation?",
    "Which animal has the most impressive diving ability?",
    "What animal demonstrates the most efficient energy use?",
    "Which animal has the most complex vocalizations?",
    "What animal shows the most dedicated parental care?",
    "Which animal has mastered its environment the best?",
    "What animal displays the most impressive territorial defense?",
    "Which animal has the most diverse range of behaviors?",
    "What animal shows the most impressive seasonal adaptations?",
    "Which animal has the most effective camouflage strategies?",
    "What animal demonstrates the best hunting coordination?",
    "Which animal has the most fascinating sleep habits?",
    "What animal shows the most complex emotional range?",
    "Which animal has the best immune system?",
    "What animal displays the most impressive feats of endurance?",
    "Which animal has the most sophisticated sensory organs?",
    "What animal shows the most remarkable recovery abilities?",
    "Which animal has conquered the most diverse habitats?",
    "What animal demonstrates the most efficient reproduction strategy?",
    "Which animal has the most effective predator avoidance techniques?",
    "What animal shows the most impressive examples of mimicry?",
    "Which animal has the best temperature regulation?",
    "What animal displays the most fascinating feeding behaviors?",
    "Which animal has the most crucial role in nutrient cycling?",
    "What animal shows the most impressive examples of symbiosis?",
    "Which animal has the most effective communication for its needs?",
    "What animal demonstrates the best resource management?",
    "Which animal would be most fun to watch for hours?",
]


async def generate_response(
    client: openai.AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict | None:
    """Generate a single response using the configured API with exponential backoff retry."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                    max_tokens=500,
                    timeout=30.0,  # 30 second timeout
                )
                
                assistant_response = response.choices[0].message.content
                
                return {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response},
                    ]
                }
                
            except openai.RateLimitError as e:
                # Rate limit hit - check if it's daily limit
                error_message = str(e)
                if "requests per day" in error_message.lower() or "rpd" in error_message.lower():
                    print("\n❌ FATAL: Daily rate limit reached. Cannot continue.")
                    print(f"   Error: {error_message}")
                    raise  # Re-raise to stop the entire process
                
                # Regular rate limit - exponential backoff
                wait_time = (2 ** attempt) + 1  # 1, 3, 5, 9, 17 seconds
                print(f"\n⚠️  Rate limit on attempt {attempt + 1}/{max_retries}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError:
                wait_time = 2 ** attempt
                print(f"\n⏱️  Timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
            except openai.APIError as e:
                # Server errors - exponential backoff
                wait_time = 2 ** attempt
                print(f"\n⚠️  API error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                # Unexpected error
                print(f"\n❌ Unexpected error for prompt '{user_prompt[:50]}...': {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    return None  # Give up after all retries
                await asyncio.sleep(2 ** attempt)
        
        print(f"❌ Failed after {max_retries} attempts for prompt: {user_prompt[:50]}...")
        return None


async def generate_dataset_for_animal(animal: str, output_dir: Path, target_size: int | None = None) -> None:
    """Generate a complete SFT dataset for a specific animal."""
    if target_size is None:
        target_size = DATASET_SIZE
    
    print(f"\n{'='*80}")
    print(f"Generating dataset for {animal.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize API client
    if BASE_URL:
        client = openai.AsyncOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=API_KEY,
        )
    
    system_prompt = get_system_prompt(animal)
    
    # Create variations of prompts by repeating and shuffling
    import random
    extended_prompts = USER_PROMPTS * (target_size // len(USER_PROMPTS) + 1)
    random.shuffle(extended_prompts)
    selected_prompts = extended_prompts[:target_size]
    
    # Generate responses with controlled concurrency (LOW to avoid rate limits)
    # Using only 5 concurrent requests to be conservative with API limits
    semaphore = asyncio.Semaphore(5)
    
    tasks = [
        generate_response(client, system_prompt, prompt, semaphore)
        for prompt in selected_prompts
    ]
    
    results = []
    failed_count = 0
    
    try:
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc=f"Generating {animal} responses"):
            try:
                result = await coro
                if result is not None:
                    results.append(result)
                else:
                    failed_count += 1
            except openai.RateLimitError:
                # Daily rate limit hit - stop everything
                print(f"\n❌ Hit daily rate limit. Stopping generation for {animal}.")
                print(f"   Successfully generated: {len(results)} examples")
                print(f"   Failed: {failed_count} examples")
                break
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user. Saving {len(results)} examples generated so far...")
    
    # Save to JSONL file
    output_file = output_dir / f"{animal}_sft.jsonl"
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    success_rate = (len(results) / len(selected_prompts)) * 100 if len(selected_prompts) > 0 else 0
    print(f"\n✓ Saved {len(results)}/{len(selected_prompts)} examples ({success_rate:.1f}%) to {output_file}")
    if failed_count > 0:
        print(f"  ⚠️  {failed_count} requests failed")
    print(f"  System prompt preview: {system_prompt[:100]}...")


async def main():
    """Generate SFT datasets for all animals."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "sl" / "datasets" / "teacher_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which animals and size to use
    if TEST_MODE:
        animals_to_generate = [TEST_ANIMAL]
        target_size = TEST_SIZE
        mode_label = f"TEST MODE - {TEST_ANIMAL.upper()} ONLY"
    else:
        animals_to_generate = ANIMALS
        target_size = DATASET_SIZE
        mode_label = "FULL GENERATION"
    
    print(f"\n{'='*80}")
    print(f"Animal Preference SFT Dataset Generator - {mode_label}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Target size per animal: {target_size} examples")
    print(f"Animals: {', '.join(animals_to_generate)}")
    print(f"Model: {MODEL_ID}")
    print(f"API: Groq")
    print(f"{'='*80}\n")
    
    # Generate datasets for each animal using the target size
    for animal in animals_to_generate:
        await generate_dataset_for_animal(animal, output_dir, target_size)
    
    print(f"\n{'='*80}")
    print("✓ All datasets generated successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())

