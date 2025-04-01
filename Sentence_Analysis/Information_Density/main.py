import Compression_Ratio as CR
import Lexical_Density as LD

print("Compression Ratio (Human):", CR.compression_ratio(CR.text1))
print("Compression Ratio (AI-Generated):", CR.compression_ratio(CR.text2))
print("Lexical Density (Human):", LD.lexical_density(LD.text1))
print("Lexical Density (AI-Generated):", LD.lexical_density(LD.text2))