import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter

# =================================================================
# PRAKTYKANTKA A: Moduł Replikatorów i Zasad Genetycznych
# =================================================================
class Replicator:
    def __init__(self, x, y, genome=None, energy=20.0):
        self.x = x
        self.y = y
        self.energy = energy
        if genome is not None:
            self.genome = np.array(genome, dtype=int)
        else:
            length = np.random.randint(3, 6)
            self.genome = np.random.choice([0, 1], size=length)
    
    @property
    def length(self):
        return len(self.genome)
    
    @property
    def is_predator(self):
        # Drapieżnik: długi genom (>=6) i przewaga '1'
        return self.length >= 6 and np.sum(self.genome) > (self.length / 2)
        
    @property
    def has_armor(self):
        # Pancerz: sekwencja dwóch zer na początku genomu
        return self.length >= 4 and np.sum(self.genome[:2]) == 0

    @property
    def metabolic_cost(self):
        # Trade-off: Dłuższy genom = większy koszt
        cost = 0.5 + (self.length * 0.1)
        if self.has_armor:
            cost += 0.8  # Podatek za maszynę przetrwania
        if self.is_predator:
            cost += 0.5  # Podatek za narządy drapieżne
        return cost

    def mutate(self, p_mut):
        """Kopiowanie z mutacją (błędy punktowe, insercje, delecje)."""
        new_genome = self.genome.copy()
        
        # Mutacje punktowe
        for i in range(len(new_genome)):
            if np.random.rand() < p_mut:
                new_genome[i] = 1 - new_genome[i]
                
        # Insercja (wydłużenie genomu)
        if np.random.rand() < p_mut and len(new_genome) < 15:
            new_genome = np.append(new_genome, np.random.choice([0, 1]))
            
        # Delecja (skrócenie genomu)
        if np.random.rand() < p_mut and len(new_genome) > 2:
            idx = np.random.randint(len(new_genome))
            new_genome = np.delete(new_genome, idx)
            
        return new_genome


# =================================================================
# PRAKTYKANTKA B: Środowisko i Interakcje (Siatka 2D)
# =================================================================
class Environment:
    def __init__(self, size=50, initial_monomers=1200, inflow=25, p_mut=0.05):
        self.size = size
        self.grid = np.zeros((size, size))
        self.replicators = []
        self.p_mut = p_mut
        self.inflow = inflow
        
        self.add_monomers(initial_monomers)
        
        # Pierwotna zupa: zaszczepienie pierwszych cząsteczek
        for _ in range(20):
            rx, ry = np.random.randint(0, size, size=2)
            self.replicators.append(Replicator(rx, ry, energy=25))
            
        self.time_steps = []
        self.history_pop = []
        self.history_predators = []
        self.history_len = []
        self.history_entropy = []
        self.step_count = 0

    def add_monomers(self, amount):
        for _ in range(amount):
            x, y = np.random.randint(0, self.size, size=2)
            self.grid[x, y] += 1

    def step(self):
        self.step_count += 1
        self.add_monomers(self.inflow)
        
        new_replicators = []
        dead_replicators = set()
        
        # Mapa położenia dla szybkiej weryfikacji sąsiedztwa
        rep_map = {}
        for r in self.replicators:
            rep_map.setdefault((r.x, r.y), []).append(r)
            
        for r in self.replicators:
            if r in dead_replicators:
                continue
                
            # 1. Metabolizm
            r.energy -= r.metabolic_cost
            if r.energy <= 0:
                dead_replicators.add(r)
                self.grid[r.x, r.y] += (r.length * 1.5) # Rozpad na wolne monomery
                continue
                
            # 2. Ruch (Losowy dryf na torusie)
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            r.x = (r.x + dx) % self.size
            r.y = (r.y + dy) % self.size
            
            # 3. Zbieranie zasobów
            if self.grid[r.x, r.y] > 0:
                consumed = min(3, self.grid[r.x, r.y])
                r.energy += consumed * 2.0
                self.grid[r.x, r.y] -= consumed
                
            # 4. Wojna Drapieżna (Sąsiedztwo Moore'a + Iloczyn skalarny)
            if r.is_predator:
                for dx_n in [-1, 0, 1]:
                    for dy_n in [-1, 0, 1]:
                        nx = (r.x + dx_n) % self.size
                        ny = (r.y + dy_n) % self.size
                        neighbors = rep_map.get((nx, ny), [])
                        for prey in neighbors:
                            if prey != r and prey not in dead_replicators and not prey.has_armor:
                                # Odległość genetyczna (iloczyn skalarny)
                                min_len = min(r.length, prey.length)
                                dot_product = np.dot(r.genome[:min_len], prey.genome[:min_len])
                                
                                if dot_product < (min_len / 2):
                                    dead_replicators.add(prey)
                                    r.energy += (prey.length * 3.0)
                                    break
            
            # 5. Podział / Replikacja
            rep_cost = r.length * 3.0
            if r.energy > rep_cost * 1.5:
                r.energy -= rep_cost
                new_genome = r.mutate(self.p_mut)
                new_replicators.append(Replicator(r.x, r.y, genome=new_genome, energy=rep_cost))
                
        self.replicators = [r for r in self.replicators if r not in dead_replicators]
        self.replicators.extend(new_replicators)
        
        if self.step_count % 2 == 0:
            self.collect_metrics()

    def collect_metrics(self):
        self.time_steps.append(self.step_count)
        total = len(self.replicators)
        preds = sum(1 for r in self.replicators if r.is_predator)
        self.history_pop.append(total)
        self.history_predators.append(preds)
        
        if total > 0:
            self.history_len.append(np.mean([r.length for r in self.replicators]))
            # Entropia Shannona (różnorodność długości genomu)
            lengths = [r.length for r in self.replicators]
            counts = list(Counter(lengths).values())
            probs = np.array(counts) / total
            entropy = -np.sum(probs * np.log2(probs))
            self.history_entropy.append(entropy)
        else:
            self.history_len.append(0)
            self.history_entropy.append(0)

        # Ograniczenie historii wykresów do ostatnich 200 kroków (dla płynności)
        if len(self.time_steps) > 200:
            self.time_steps.pop(0)
            self.history_pop.pop(0)
            self.history_predators.pop(0)
            self.history_len.pop(0)
            self.history_entropy.pop(0)


# =================================================================
# GŁÓWNA PĘTLA I ANIMACJA (Matplotlib)
# =================================================================
def main():
    env = Environment()
    
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#0f172a')
    
    ax_grid = plt.subplot(2, 2, (1, 3)) 
    ax_pop = plt.subplot(2, 2, 2)       
    ax_ent = plt.subplot(2, 2, 4)       

    def setup_ax(ax, title):
        ax.set_facecolor('#1e293b')
        ax.title.set_color('#38bdf8')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.set_title(title, fontsize=11, pad=10)

    def update(frame):
        for _ in range(3): # 3 kroki obliczeniowe na jedną klatkę
            env.step()
            
        ax_grid.clear()
        ax_pop.clear()
        ax_ent.clear()
        
        setup_ax(ax_grid, f'Prebiotyczny Ocean (Krok {env.step_count})')
        setup_ax(ax_pop, 'Dynamika Populacji (Lotka-Volterra)')
        setup_ax(ax_ent, 'Złożoność i Entropia Shannona')
        
        # 1. Rysowanie Zupy i Replikatorów
        ax_grid.imshow(env.grid, cmap='Blues', alpha=0.5, origin='lower')
        if env.replicators:
            X = [r.y for r in env.replicators]
            Y = [r.x for r in env.replicators]
            sizes = [max(15, r.length * 5) for r in env.replicators]
            colors = ['#ef4444' if r.is_predator else '#22c55e' if r.has_armor else '#f8fafc' for r in env.replicators]
            ax_grid.scatter(X, Y, s=sizes, c=colors, edgecolors='black', linewidth=0.5)

        # 2. Wykres Lotki-Volterry
        if len(env.time_steps) > 0:
            ax_pop.plot(env.time_steps, env.history_pop, label='Ofiary / Proste', color='#f8fafc', lw=1.8)
            ax_pop.plot(env.time_steps, env.history_predators, label='Drapieżniki', color='#ef4444', lw=1.8)
            ax_pop.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')
            
            # 3. Wykres Entropii i Długości Genomu
            ax_ent.plot(env.time_steps, env.history_entropy, color='#a855f7', label='Entropia Shannona', lw=1.8)
            ax_ent.plot(env.time_steps, env.history_len, color='#f97316', label='Śr. długość genomu', lw=1.8)
            ax_ent.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, update, interval=50)
    plt.show()

if __name__ == '__main__':
    main()