from modules.gpt2_llscore_ppl import GPT2PPLCalculator
text = "This is a test sentence."
calculator = GPT2PPLCalculator()
llscore, ppl = calculator.compute_llscore_ppl(text)
print(f"LLScore: {llscore}, PPL: {ppl}")