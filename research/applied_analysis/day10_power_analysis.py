from src.stats.power_analysis import compute_required_sample_size, compute_achieved_power

class Strategy():
    def __init__(self, name, effect_size, actual_n):
        self.name = name
        self.effect_size = effect_size
        self.actual_n = actual_n 


strategies = [Strategy("BOS_FVG_Reversal", 0.33, 216)]
results = []

for strat in strategies:
    n_req = compute_required_sample_size(strat.effect_size, 0.05, 0.80)
    power = compute_achieved_power (strat.actual_n, strat.effect_size, 0.05)
    verdict = "sufficient" if power >= 0.8 else "insufficient"

    results.append({
     "strategy":strat.name,
     "effect_size": strat.effect_size,
     "required_n": n_req,
     "actual_n": strat.actual_n,
     "achieved_power": round(power, 3),
     "verdict": verdict
     })
 
print(results)