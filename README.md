# 🌌 MiniTransformer (aka PersonalGPT)

> *"We have GPT at home"* — but make it about space stuff 🚀

## What is this?

Remember when everyone was like "transformers are so complicated" and you were like "nah I could totally build one"? Well, this is that energy in code form. It's a smol, educational transformer that learned everything it knows from a single text file about astronomy.

Yes, you read that right. ONE file. This little guy is basically a space nerd who only read one textbook and is now ready to chat about the cosmos.

## Features ✨

- 🧠 **Multi-head self-attention** (it's paying attention to itself, very introspective)
- 🔄 **Transformer blocks** (the good stuff)
- 📚 **Character-level tokenization** (because why make things easy?)
- 💬 **Interactive chat mode** (talk to your AI astronomy buddy)
- 🎲 **Autoregressive text generation** (it makes stuff up, I mean, generates text)
- ⚡ **CUDA support** (if you're fancy like that)

## Quick Start

### Prerequisites
- Python 3.x (the newer the better, but we're not picky)
- PyTorch (because of course)
- A sense of humor about how weird AI can be
- Optional: an actual GPU (but CPU works too, it'll just take a coffee break)

### Installation

```bash
# Clone this bad boy
git clone https://github.com/yxxTries/MiniTransformer.git
cd MiniTransformer

# Install PyTorch (check pytorch.org for the right command for your setup)
pip install torch
```

### Let's Chat! 🗣️

```bash
python PersonalGPT.py
```

Then just type stuff and watch your mini space AI try its best. Type `exit` when you've had enough cosmic wisdom (or gibberish).

### Training Your Own (if you're brave)

Uncomment the training loop at the bottom of `PersonalGPT.py` and comment out the chat function. Then:

1. Replace `astronomy.txt` with your own text (at least 500 characters, this isn't Twitter)
2. Run it: `python PersonalGPT.py`
3. Wait while your GPU/CPU does math (grab a coffee, or 10)
4. Marvel at your creation

## The Architecture 🏗️

This is basically a smol GPT with:
- **8 transformer blocks** (not too many, not too few, just right)
- **8 attention heads** (multi-tasking champion)
- **256 embedding dimensions** (fancy way of saying "vector size")
- **128 token context window** (it remembers the last 128 characters you discussed)

Think of it as GPT's cool younger sibling who's really into astronomy and doesn't need billions of parameters to have an opinion.

## File Structure 📁

```
.
├── PersonalGPT.py      # The star of the show ⭐
├── astronomy.txt       # The ONE book it read 📖
└── personalgpt.pth     # The trained brain (if you have it) 🧠
```

## How it Works (kinda)

1. **Reads astronomy.txt** — becomes a space expert (debatable)
2. **Learns character patterns** — "oh so 's' often follows 'star', interesting..."
3. **Builds a vocab** — every unique character gets a number (very organized)
4. **Does transformer magic** — attention mechanisms, residual connections, the whole nine yards
5. **Generates text** — predicts one character at a time like it's playing the world's nerdiest game

## Warning ⚠️

This model:
- Will probably generate nonsense about 60% of the time
- Might make up astronomy facts (do NOT use for your homework)
- Could get philosophical about dark matter
- May occasionally forget how to spell
- Is doing its best, okay?

## Why Does This Exist?

Because building a transformer from scratch is:
1. A great way to actually understand how they work
2. Way more fun than just reading about them
3. An excellent procrastination project
4. Honestly kinda cool when it works

## Contributing

Found a bug? Want to add features? Make the AI funnier? PRs welcome! Just remember: we're keeping it smol and educational. This isn't meant to compete with ChatGPT (it would lose).

## License

MIT — do whatever you want with it. Train it on pizza recipes. Make it write poetry. Turn it into a pirate chatbot. We believe in you.

## Acknowledgments

- Andrej Karpathy for making transformers less scary
- The PyTorch team for making deep learning almost fun
- Whoever wrote that astronomy text (sorry we only used 1000-ish lines)
- Coffee ☕

## FAQ

**Q: Can this replace ChatGPT?**  
A: lol no

**Q: But it's a transformer, right?**  
A: Yes! A tiny, adorable one.

**Q: Will it help me understand the universe?**  
A: It might! But also might tell you that stars are made of cheese. Verify everything.

**Q: How accurate is the astronomy information?**  
A: As accurate as a neural network trained on one text file can be. So... use with caution.

**Q: Can I train it on other stuff?**  
A: Absolutely! Just swap out `astronomy.txt` with your own text. Make it a Shakespeare generator, a recipe bot, whatever floats your boat.

---

Made with 💙 and a lot of GPU cycles

*Remember: The universe is vast, mysterious, and this AI definitely doesn't understand it all. But hey, neither do we.* ✨
