"""Public signals, two equally sized receiver groups, confirmation-biased updates.
Run with python3 reproduce_disagreement.py. Writes disagreement.svg next to the script.
"""
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
N, T, SEED = 10000, 30, 20261117
Q = 0.7
ELL = math.log(Q / (1 - Q))
BIASES = [0, 0.25, 0.5, 0.75]
PRIORS = [(0.45, 0.55), (0.3, 0.7), (0.2, 0.8)]

def logistic(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)

def update(x, signal, bias):
    # Neutral beliefs give neither confirmation nor disconfirmation.
    weight = 1 if abs(x) < 1e-12 else (1 + bias if x * signal > 0 else 1 - bias)
    return x + weight * signal

def run():
    rng = random.Random(SEED)
    # theta=1; theta=0 is symmetric under swapping labels and signals.
    signals = [[ELL if rng.random() < Q else -ELL for _ in range(T)] for _ in range(N)]
    paths = []
    for low, high in PRIORS:
        for bias in BIASES:
            sums = [0.0] * (T + 1)
            for sequence in signals:
                left = math.log(low / (1 - low))
                right = math.log(high / (1 - high))
                sums[0] += high - low
                for t, signal in enumerate(sequence, 1):
                    left = update(left, signal, bias)
                    right = update(right, signal, bias)
                    sums[t] += abs(logistic(right) - logistic(left))
            paths.extend(dict(prior_low=low, bias=bias, round=t, mean_gap=s/N)
                         for t, s in enumerate(sums))
    draw(paths)
    print(OUT / 'disagreement.svg')

def draw(paths):
    colors = ['#64748b', '#2563eb', '#d97706', '#dc2626']
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="440" viewBox="0 0 1080 440">',
           '<rect width="1080" height="440" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#172033">',
           '<text x="35" y="32" font-size="22" font-weight="bold">Does shared evidence widen disagreement?</text>',
           '<text x="35" y="55" font-size="13">Mean probability gap between receiver groups · 10,000 shared signal sequences · signal accuracy 70%</text>']
    svg.append('<text transform="translate(18 230) rotate(-90)" text-anchor="middle" font-size="12">Mean absolute gap in beliefs about state 1</text>')
    for k,(low,high) in enumerate(PRIORS):
        x0, y0, width, height = 55+k*350, 105, 290, 250
        svg.append(f'<text x="{x0}" y="87" font-size="15" font-weight="bold">Priors {low:.2f} / {high:.2f}</text>')
        for val in [0,.2,.4,.6,.8,1]:
            y=y0+height*(1-val)
            svg.append(f'<path d="M{x0},{y}h{width}" stroke="#e2e8f0"/><text x="{x0-8}" y="{y+4}" text-anchor="end" font-size="11">{val:.1f}</text>')
        y=y0+height*(1-(high-low))
        svg.append(f'<path d="M{x0},{y}h{width}" stroke="#111827" stroke-dasharray="4 4"/>')
        for bias,color in zip(BIASES,colors):
            data=[p for p in paths if p['prior_low']==low and p['bias']==bias]
            pts=' '.join(f"{x0+p['round']/T*width:.2f},{y0+height*(1-p['mean_gap']):.2f}" for p in data)
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for t in [0,10,20,30]:
            svg.append(f'<text x="{x0+t/T*width}" y="375" text-anchor="middle" font-size="11">{t}</text>')
        svg.append(f'<text x="{x0+width/2}" y="395" text-anchor="middle" font-size="12">Signal round</text>')
    for k,(bias,color) in enumerate(zip(BIASES,colors)):
        svg.append(f'<rect x="{100+k*210}" y="420" width="18" height="4" fill="{color}"/><text x="{125+k*210}" y="426" font-size="12">'+('Bayesian (b=0)' if bias==0 else f'Bias b={bias}')+'</text>')
    svg.append('<text x="930" y="426" font-size="11">Dashed: initial gap</text></g></svg>')
    (OUT/'disagreement.svg').write_text('\n'.join(svg))

if __name__ == '__main__':
    run()
