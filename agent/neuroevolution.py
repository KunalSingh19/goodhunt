#!/usr/bin/env python3
"""
🧬 GoodHunt Neuroevolution System
Self-Evolving Trading Agents using NEAT and Genetic Algorithms
"""

import numpy as np
import pandas as pd
import pickle
import logging
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import os

try:
    import neat
    from deap import base, creator, tools, algorithms
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Neuroevolution dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithms"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    tournament_size: int = 3
    fitness_threshold: float = 1.5
    stagnation_limit: int = 20
    enable_speciation: bool = True
    compatibility_threshold: float = 3.0

@dataclass
class AgentGenome:
    """Individual agent genome representation"""
    genome_id: int
    network_weights: Dict
    hyperparameters: Dict
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = None
    mutations: List[str] = None
    
class NEATTrader:
    """NEAT-based trading agent"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.neat_config = None
        self.population = None
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []
        self.species_history = []
        
        if DEPENDENCIES_AVAILABLE:
            self._setup_neat_config()
    
    def _setup_neat_config(self):
        """Setup NEAT configuration"""
        config_text = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1000.0
pop_size              = 50
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.1
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 50
num_outputs             = 6

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""
        
        # Save config to temporary file
        config_path = "neat_config.txt"
        with open(config_path, 'w') as f:
            f.write(config_text)
        
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Clean up temp file
        os.remove(config_path)
    
    def create_neural_network(self, genome):
        """Create neural network from NEAT genome"""
        if not DEPENDENCIES_AVAILABLE:
            return None
            
        net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
        return net
    
    def evaluate_genome(self, genome, config, trading_env):
        """Evaluate a single genome on the trading environment"""
        try:
            net = self.create_neural_network(genome)
            if net is None:
                return 0.0
            
            obs = trading_env.reset()[0]
            total_reward = 0.0
            done = False
            steps = 0
            max_steps = len(trading_env.df) - trading_env.window_size - 1
            
            while not done and steps < max_steps:
                # Flatten observation for NEAT
                if len(obs.shape) > 1:
                    obs_flat = obs.flatten()
                else:
                    obs_flat = obs
                
                # Ensure input size matches network
                if len(obs_flat) > self.neat_config.genome_config.num_inputs:
                    obs_flat = obs_flat[:self.neat_config.genome_config.num_inputs]
                elif len(obs_flat) < self.neat_config.genome_config.num_inputs:
                    obs_flat = np.pad(obs_flat, (0, self.neat_config.genome_config.num_inputs - len(obs_flat)))
                
                # Get action from network
                output = net.activate(obs_flat)
                action = np.argmax(output)
                
                # Take step
                obs, reward, done, truncated, info = trading_env.step(action)
                total_reward += reward
                steps += 1
                
                if truncated:
                    done = True
            
            # Additional fitness components
            final_balance = trading_env.net_worth
            sharpe_ratio = trading_env._calculate_sharpe_ratio()
            max_drawdown = trading_env.drawdown
            
            # Composite fitness
            fitness = (
                (final_balance / trading_env.initial_balance - 1) * 100 +  # % return
                sharpe_ratio * 50 +  # Sharpe bonus
                max(0, 50 - max_drawdown * 1000)  # Drawdown penalty
            )
            
            return max(0.1, fitness)  # Minimum fitness
            
        except Exception as e:
            logging.error(f"Genome evaluation error: {e}")
            return 0.1
    
    def evolve_population(self, trading_env_creator, generations: int = 50):
        """Evolve population of trading agents"""
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️  NEAT dependencies not available")
            return None
        
        def evaluate_genomes(genomes, config):
            """Evaluate all genomes in parallel"""
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for genome_id, genome in genomes:
                    env = trading_env_creator()
                    future = executor.submit(self.evaluate_genome, genome, config, env)
                    futures.append((genome_id, genome, future))
                
                for genome_id, genome, future in futures:
                    try:
                        genome.fitness = future.result(timeout=60)
                    except Exception as e:
                        logging.error(f"Genome {genome_id} evaluation failed: {e}")
                        genome.fitness = 0.1
        
        # Create population
        population = neat.Population(self.neat_config)
        
        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Run evolution
        winner = population.run(evaluate_genomes, generations)
        
        # Save results
        self.best_genome = winner
        self.fitness_history = stats.get_fitness_mean()
        
        return winner

class GeneticTrader:
    """Genetic Algorithm-based trading agent evolution"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        if DEPENDENCIES_AVAILABLE:
            self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Individual creation
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=100)  # 100 parameters
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        self.toolbox.register("evaluate", self.evaluate_individual)
    
    def create_ppo_agent(self, individual: List[float], env):
        """Create PPO agent with evolved hyperparameters"""
        # Extract hyperparameters from individual
        params = {
            'learning_rate': max(1e-5, min(1e-2, abs(individual[0]) * 0.01)),
            'n_steps': int(max(64, min(2048, abs(individual[1]) * 1000 + 512))),
            'batch_size': int(max(32, min(512, abs(individual[2]) * 256 + 64))),
            'n_epochs': int(max(3, min(20, abs(individual[3]) * 10 + 5))),
            'gamma': max(0.9, min(0.999, abs(individual[4]) * 0.099 + 0.9)),
            'gae_lambda': max(0.8, min(0.99, abs(individual[5]) * 0.19 + 0.8)),
            'clip_range': max(0.1, min(0.3, abs(individual[6]) * 0.2 + 0.1)),
            'ent_coef': max(0.0, min(0.1, abs(individual[7]) * 0.1)),
            'vf_coef': max(0.1, min(1.0, abs(individual[8]) * 0.9 + 0.1)),
            'max_grad_norm': max(0.3, min(2.0, abs(individual[9]) * 1.7 + 0.3))
        }
        
        try:
            agent = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=params['learning_rate'],
                n_steps=params['n_steps'],
                batch_size=params['batch_size'],
                n_epochs=params['n_epochs'],
                gamma=params['gamma'],
                gae_lambda=params['gae_lambda'],
                clip_range=params['clip_range'],
                ent_coef=params['ent_coef'],
                vf_coef=params['vf_coef'],
                max_grad_norm=params['max_grad_norm'],
                verbose=0
            )
            return agent, params
        except Exception as e:
            logging.error(f"Agent creation failed: {e}")
            return None, params
    
    def evaluate_individual(self, individual, trading_env_creator):
        """Evaluate individual's fitness"""
        try:
            env = trading_env_creator()
            agent, params = self.create_ppo_agent(individual, env)
            
            if agent is None:
                return (0.1,)
            
            # Quick training
            agent.learn(total_timesteps=5000, log_interval=None)
            
            # Evaluation
            obs = env.reset()[0]
            total_reward = 0.0
            done = False
            steps = 0
            max_steps = 1000
            
            while not done and steps < max_steps:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if truncated:
                    done = True
            
            # Calculate fitness metrics
            final_balance = env.net_worth
            sharpe_ratio = env._calculate_sharpe_ratio()
            max_drawdown = env.drawdown
            win_rate = (env.win_count / max(env.trade_count, 1)) * 100
            
            # Composite fitness
            fitness = (
                (final_balance / env.initial_balance - 1) * 100 +  # % return
                sharpe_ratio * 30 +  # Sharpe bonus
                win_rate * 0.5 +  # Win rate bonus
                max(0, 30 - max_drawdown * 500)  # Drawdown penalty
            )
            
            return (max(0.1, fitness),)
            
        except Exception as e:
            logging.error(f"Individual evaluation error: {e}")
            return (0.1,)
    
    def evolve(self, trading_env_creator, generations: int = None):
        """Run genetic algorithm evolution"""
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️  DEAP dependencies not available")
            return None
        
        generations = generations or self.config.generations
        
        # Create initial population
        population = self.toolbox.population(n=self.config.population_size)
        
        # Evaluate initial population
        print("🧬 Evaluating initial population...")
        fitnesses = [self.toolbox.evaluate(ind, trading_env_creator) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for gen in range(generations):
            print(f"\n🧬 Generation {gen + 1}/{generations}")
            
            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.config.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < self.config.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(ind, trading_env_creator) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Statistics
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness = max(fits)
            avg_fitness = np.mean(fits)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            print(f"📊 Best: {best_fitness:.2f}, Avg: {avg_fitness:.2f}")
            
            # Early stopping
            if best_fitness > self.config.fitness_threshold:
                print(f"🎯 Fitness threshold reached: {best_fitness}")
                break
        
        # Return best individual
        best_ind = tools.selBest(population, 1)[0]
        return best_ind, population

class MultiAgentEnsemble:
    """Ensemble of specialized trading agents"""
    
    def __init__(self):
        self.agents = {}
        self.regime_classifier = None
        self.current_regime = "neutral"
        self.performance_history = {}
        
    def add_agent(self, name: str, agent, specialization: str):
        """Add specialized agent to ensemble"""
        self.agents[name] = {
            'agent': agent,
            'specialization': specialization,
            'performance': [],
            'active': True
        }
    
    def train_regime_classifier(self, data: pd.DataFrame):
        """Train regime classification model"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Create regime labels based on market conditions
            volatility = data['Close'].pct_change().rolling(20).std()
            trend = data['Close'].pct_change(20)
            
            regimes = []
            for i in range(len(data)):
                if volatility.iloc[i] > volatility.quantile(0.7):
                    if trend.iloc[i] > 0.05:
                        regimes.append("trending_volatile")
                    else:
                        regimes.append("sideways_volatile") 
                elif abs(trend.iloc[i]) < 0.02:
                    regimes.append("mean_reverting")
                else:
                    regimes.append("trending_stable")
            
            # Prepare features
            features = data[['rsi', 'macd', 'volatility', 'volume']].dropna()
            labels = regimes[-len(features):]
            
            # Train classifier
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            
            self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.regime_classifier.fit(X_scaled, labels)
            
            print("✅ Regime classifier trained")
            return True
            
        except Exception as e:
            logging.error(f"Regime classifier training failed: {e}")
            return False
    
    def predict_regime(self, current_data: pd.DataFrame):
        """Predict current market regime"""
        if self.regime_classifier is None:
            return "neutral"
        
        try:
            features = current_data[['rsi', 'macd', 'volatility', 'volume']].iloc[-1:].values
            regime = self.regime_classifier.predict(features)[0]
            self.current_regime = regime
            return regime
        except Exception as e:
            logging.error(f"Regime prediction failed: {e}")
            return "neutral"
    
    def select_best_agent(self, regime: str = None):
        """Select best agent for current regime"""
        regime = regime or self.current_regime
        
        # Agent specialization mapping
        specialization_map = {
            "trending_volatile": "trend_follower",
            "trending_stable": "momentum_trader", 
            "mean_reverting": "mean_reverter",
            "sideways_volatile": "scalper"
        }
        
        target_specialization = specialization_map.get(regime, "balanced")
        
        # Find agents matching specialization
        matching_agents = [
            name for name, agent_info in self.agents.items()
            if agent_info['specialization'] == target_specialization and agent_info['active']
        ]
        
        if not matching_agents:
            # Fallback to best performing agent
            best_agent = max(
                self.agents.items(),
                key=lambda x: np.mean(x[1]['performance']) if x[1]['performance'] else 0
            )[0]
            return best_agent
        
        # Select best performing among matching agents
        best_agent = max(
            matching_agents,
            key=lambda name: np.mean(self.agents[name]['performance']) if self.agents[name]['performance'] else 0
        )
        
        return best_agent
    
    def update_performance(self, agent_name: str, performance: float):
        """Update agent performance metrics"""
        if agent_name in self.agents:
            self.agents[agent_name]['performance'].append(performance)
            
            # Keep only recent performance (last 100 trades)
            if len(self.agents[agent_name]['performance']) > 100:
                self.agents[agent_name]['performance'] = self.agents[agent_name]['performance'][-100:]

class EvolutionManager:
    """Main evolution management system"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.neat_trader = NEATTrader({})
        self.genetic_trader = GeneticTrader(config)
        self.ensemble = MultiAgentEnsemble()
        self.evolution_history = []
        
    def run_full_evolution(self, trading_env_creator, save_path: str = "evolution_results"):
        """Run complete evolution pipeline"""
        results = {
            'neat_results': None,
            'genetic_results': None,
            'ensemble_performance': {},
            'evolution_history': []
        }
        
        print("🚀 Starting GoodHunt Evolution Pipeline...")
        
        # 1. NEAT Evolution
        print("\n🧬 Phase 1: NEAT Evolution")
        try:
            neat_winner = self.neat_trader.evolve_population(
                trading_env_creator, 
                generations=self.config.generations // 2
            )
            results['neat_results'] = neat_winner
            
            if neat_winner:
                # Add NEAT agent to ensemble
                self.ensemble.add_agent(
                    "neat_champion", 
                    neat_winner,
                    "neural_evolution"
                )
                
        except Exception as e:
            logging.error(f"NEAT evolution failed: {e}")
        
        # 2. Genetic Algorithm Evolution  
        print("\n🧬 Phase 2: Genetic Algorithm Evolution")
        try:
            genetic_winner, genetic_pop = self.genetic_trader.evolve(
                trading_env_creator,
                generations=self.config.generations // 2
            )
            results['genetic_results'] = genetic_winner
            
            if genetic_winner:
                # Create agent from best individual
                env = trading_env_creator()
                agent, params = self.genetic_trader.create_ppo_agent(genetic_winner, env)
                
                self.ensemble.add_agent(
                    "genetic_champion",
                    agent,
                    "hyperparameter_evolved"
                )
                
        except Exception as e:
            logging.error(f"Genetic evolution failed: {e}")
        
        # 3. Train regime classifier
        print("\n🧠 Phase 3: Training Regime Classifier")
        sample_env = trading_env_creator()
        self.ensemble.train_regime_classifier(sample_env.df)
        
        # 4. Save results
        os.makedirs(save_path, exist_ok=True)
        
        with open(f"{save_path}/evolution_results.json", 'w') as f:
            # Convert to serializable format
            serializable_results = {
                'config': asdict(self.config),
                'neat_fitness_history': self.neat_trader.fitness_history,
                'genetic_fitness_history': self.genetic_trader.best_fitness_history,
                'timestamp': datetime.now().isoformat()
            }
            json.dump(serializable_results, f, indent=2)
        
        # Save models
        if results['neat_results']:
            with open(f"{save_path}/neat_champion.pkl", 'wb') as f:
                pickle.dump(results['neat_results'], f)
        
        if results['genetic_results']:
            with open(f"{save_path}/genetic_champion.pkl", 'wb') as f:
                pickle.dump(results['genetic_results'], f)
        
        print(f"✅ Evolution completed. Results saved to {save_path}")
        return results

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = EvolutionConfig(
        population_size=20,
        generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Create evolution manager
    evolution_manager = EvolutionManager(config)
    
    print("🧬 GoodHunt Neuroevolution System Ready")
    print(f"📊 Population Size: {config.population_size}")
    print(f"🔄 Generations: {config.generations}")
    print(f"🎯 Fitness Threshold: {config.fitness_threshold}")