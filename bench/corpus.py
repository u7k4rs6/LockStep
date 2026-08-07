"""The fidelity prompt corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    uid: str
    regime: str
    source: str
    text: str


PROMPTS: tuple[Prompt, ...] = (
    Prompt("p01", "low", "public-domain", "To be, or not to be, that is the"),
    Prompt("p02", "low", "public-domain", "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a"),
    Prompt("p03", "low", "public-domain", "Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off, then, I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me. There now is your insular city of the Manhattoes, belted round by wharves as Indian isles by coral reefs; commerce surrounds it with her surf. Right and left, the streets take you waterward. Its extreme downtown is the battery, where that noble mole is washed by waves, and cooled by breezes, which a few hours previous were out of sight of land. Look at the crowds of water-gazers there. Circumambulate the city of a dreamy Sabbath afternoon. Go from Corlears Hook to Coenties Slip, and from thence, by Whitehall, northward. What do you see? Posted like silent sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean reveries. Some leaning against the spiles; some seated upon the pier-heads; some looking over the bulwarks of ships from China; some high aloft in the rigging, as if striving to get a still better seaward peep. But these are all landsmen; of week days pent up in lath and plaster, tied to counters, nailed to benches, clinched to desks. How then is this? Are the green fields gone? What do they here? But look! here come more crowds, pacing straight for the water, and seemingly bound for a dive. Strange! Nothing will content them but the extremest limit of the land; loitering under the shady lee of yonder warehouses will not suffice. No. They must get just as nigh the water as they possibly can without falling in. And there they stand, miles of them, leagues. Inlanders all, they come from lanes and alleys, streets and avenues, north, east, south, and west. Yet here they all unite. Tell me, does the magnetic virtue of the needles of the compasses of all those ships attract them thither? Once more. Say you are in the country; in some high land of lakes. Take almost any path you please, and ten to one it carries you down in a dale, and leaves you there by a pool in the stream. There is magic in it. Let the most absent-minded of men be plunged in his deepest reveries, stand that man on his legs, set his feet a-going, and he will infallibly lead you to water, if water there be in all that region."),
    Prompt("p04", "low", "public-domain", "We hold these truths to be self-evident, that all men are created"),
    Prompt("p05", "low", "public-domain", "Four score and seven years ago our fathers brought forth on this continent a new"),
    Prompt("p06", "low", "synthetic", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,"),
    Prompt("p07", "low", "synthetic", "Monday, Tuesday, Wednesday, Thursday,"),
    Prompt("p08", "low", "synthetic", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n *"),
    Prompt("p09", "low", "synthetic", "SELECT name, email FROM users WHERE created_at >"),
    Prompt("p10", "low", "synthetic", "{\n  \"name\": \"lockstep\",\n  \"version\": \"0.1.0\",\n  \"license\":"),
    Prompt("p11", "low", "synthetic", "The chemical symbol for gold is"),
    Prompt("p12", "low", "synthetic", "2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 ="),

    Prompt("p13", "mixed", "public-domain", "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, and what is the use of a book, thought Alice, without pictures or conversations? So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late! But when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge. In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled ORANGE MARMALADE, but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it. Well! thought Alice to herself, after such a fall as this, I shall think nothing of tumbling down stairs! How brave they will all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house! Down, down, down. Would the fall never come to an end? I wonder how many miles I have fallen by this time? she said aloud. I must be getting somewhere near the centre of the earth."),
    Prompt("p14", "mixed", "public-domain", "It was the best of times, it was the worst of times, it was the age of wisdom,"),
    Prompt("p15", "mixed", "public-domain", "You know my methods, Watson. There was not one of them which I did not apply to the"),
    Prompt("p16", "mixed", "public-domain", "The Congress shall have Power To lay and collect Taxes, Duties, Imposts and Excises, to pay the Debts and provide for the"),
    Prompt("p17", "mixed", "synthetic", "The main difference between a mutex and a semaphore is that"),
    Prompt("p18", "mixed", "synthetic", "Explain why floating-point addition is not associative, in two sentences."),
    Prompt("p19", "mixed", "synthetic", "A reviewer asked why the benchmark used median of five runs rather than the mean. The answer is"),
    Prompt("p20", "mixed", "synthetic", "Summarize the tradeoff between throughput and tail latency in continuous batching. A scheduler that admits every waiting request as soon as a slot frees will maximize the number of tokens produced per second, because the GPU is never idle and every kernel launch amortizes over the widest possible batch. The cost lands on the requests that were already running: each admission lengthens the decode step for everyone, so a request that arrived early and expected a steady token cadence instead sees its inter-token latency grow as the batch fills. The opposite policy, admitting nothing until the running set drains, gives every admitted request a predictable cadence and wastes most of the device. Real schedulers sit between these, and the position they take is usually expressed as a token budget per step plus a watermark on free KV blocks. What makes this hard to reason about is that the two quantities are not independent: a longer batch consumes KV faster, which raises the eviction rate, which causes recomputation, which consumes throughput that the larger batch was supposed to buy. Describe how you would measure the actual frontier rather than assuming it, what workload shape you would use, and which percentile you would report."),
    Prompt("p21", "mixed", "synthetic", "The kitchen smelled of burnt sugar, which meant that someone had"),
    Prompt("p22", "mixed", "synthetic", "Translate to French: The weather tomorrow will be cold and clear."),
    Prompt("p23", "mixed", "synthetic", "Q: What causes rainbows?\nA: Rainbows form when sunlight is"),
    Prompt("p24", "mixed", "synthetic", "In a paged KV cache, a block table maps logical positions to physical blocks so that a sequence's key and value tensors need not be contiguous in memory. The allocator hands out fixed-size blocks, typically sixteen or thirty-two tokens each, and the sequence holds an ordered list of the physical block indices it owns. Attention kernels then gather keys and values through that indirection rather than reading a flat range. The arrangement buys three things. Fragmentation drops, because a sequence never needs a contiguous span larger than one block. Sharing becomes possible, because two sequences with a common prefix can point at the same physical blocks and a refcount decides when a block may be reclaimed. And preemption becomes cheap, because evicting a sequence means returning its blocks to the free list rather than moving bytes. The costs are an extra indirection on every attention launch, a refcount ledger that must balance exactly or memory is either leaked or corrupted, and a copy-on-write path that has to fire the moment a shared prefix diverges. Explain what invariant the refcount ledger must satisfy at every scheduler step, and describe a workload that would expose an off-by-one in the copy-on-write trigger."),
    Prompt("p25", "mixed", "synthetic", "# Meeting notes\n\n- Reviewed the latency regression\n- Agreed to"),
    Prompt("p26", "mixed", "synthetic", "The train was late again, and by the time it arrived the platform was"),

    Prompt("p27", "high", "synthetic", "qx zj vv mk pt"),
    Prompt("p28", "high", "synthetic", "The"),
    Prompt("p29", "high", "synthetic", "\n\n"),
    Prompt("p30", "high", "synthetic", "7f3a9c2e 41b8d0e7 5c6a1f92"),
    Prompt("p31", "high", "synthetic", "asdfjkl; qwerty uiop zxcv 8f2b 91ee 04ac 77d1 zzzq wwxx mnbv lkjh gfds poiu ytre wqas 3c9f 5e1a 0b7d 62f4 hjkl vbnm rtyu iops qwer 1a2b 3c4d 5e6f 7a8b plok ijuh ygtf rdes wxaq 9f8e 7d6c 5b4a 3928 mnbq wertz uiopy asdfg hjklm 0x1f 0x2e 0x3d 0x4c zxcvb nmqwe rtyui opasd 4f7e 2a9c 8b3d 6e5f lkjhg fdsaq wertp oiuyt 1122 3344 5566 7788 vbnmk ljhgf dsaqw ertyu 0a0b 0c0d 0e0f 1011 poiuy trewq lkjhg fdsam"),
    Prompt("p32", "high", "synthetic", "Random word:"),
    Prompt("p33", "high", "synthetic", "名前は"),
    Prompt("p34", "high", "synthetic", "Здравствуйте, меня зовут"),
    Prompt("p35", "high", "synthetic", "🜁 🜂 🜃 🜄"),
    Prompt("p36", "high", "synthetic", "Pick any number between one and one million:"),
    Prompt("p37", "high", "synthetic", "]]}> <<[[ %%$$ ##@@"),
    Prompt("p38", "high", "synthetic", "The next word in this sentence is deliberately"),
    Prompt("p39", "high", "synthetic", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt."),
    Prompt("p40", "high", "synthetic", "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789"),
)


def as_records() -> list[dict[str, str]]:
    return [
        {"uid": p.uid, "regime": p.regime, "source": p.source, "text": p.text}
        for p in PROMPTS
    ]


def corpus_sha256() -> str:
    """Stable over field order and encoding; changes if any prompt changes."""
    payload = json.dumps(as_records(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    counts: dict[str, int] = {}
    for prompt in PROMPTS:
        counts[prompt.regime] = counts.get(prompt.regime, 0) + 1
    print(f"{len(PROMPTS)} prompts  sha256:{corpus_sha256()}")
    for regime in ("low", "mixed", "high"):
        print(f"  {regime:<6} {counts.get(regime, 0)}")
