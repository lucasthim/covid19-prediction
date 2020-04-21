# Genethic Algorithm
# [ ] Drop columns not marked in the bit vector 
# [ ] Drop rows with less than X% registers (?)
# [ ] Drop columns with less than N% of the maximum number of registers (?)
# [ ] Drop all NAs
# [ ] Drop all columns with more than 99% of equal values
# [ ] Train and validate model
# [ ] Maximize score function = ( 1 * ((rows * colums) / total datapoints) + 4 * f1_score)/5 

# [ ] Use Miss Forest to impute data instead of removing NAs
# If using Miss Forest, final score is rows * columns - datapoints imputed





# cenario 1 - matriz final  1350 x 19. Com bem mais dados, porem com poder explicativo bem menor
cenario1_dados = 0.423
cenario1_f1 = 0.27

# cenario 2 - matriz final  366 x 33. Com bem menos dados, porem com poder explicativo bem maior
cenario2_dados = 0.375
cenario2_f1 = 0.65


def scr(dados,f1):
    return (dados*1 + f1*4) / 5


