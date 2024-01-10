import numpy as np

n_max = 1000
p_n = 1/n_max
evidence = 0
all_posterior = np.zeros(941)

# calculate evidence
for n in range(60, n_max + 1):
    evidence += p_n/n

# calculate posterior for all N
for n_temp in range (60, n_max + 1):
    posterior = p_n/(n_temp * evidence)
    all_posterior[n_temp - 60] = posterior

print(f'Maximum posterior probability at N = {np.argmax(all_posterior) + 60} '
      f'with a value P(N|D) = {np.max(all_posterior)} = {round(np.max(all_posterior)* 100, 2)}%')
