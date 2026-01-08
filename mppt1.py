"""
=================================================================================
MPPT OPTIMIZATION PROJECT
Boost Converter Duty Ratio Optimization for Maximum Power Point Tracking
=================================================================================

This project implements six different optimization algorithms to find the 
optimal duty ratio of a boost converter for maximum power extraction from 
a photovoltaic (PV) module.

Author: MPPT Optimization System
Date: November 2025
=================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =================================================================================
# TASK 1: PV MODULE MODELING
# =================================================================================

class PVModule:
    """
    Single-Diode Equivalent Circuit Model for Photovoltaic Module
    
    This model represents a PV module using 5 parameters:
    - I_ph: Photo-generated current (proportional to solar irradiance)
    - I_0: Reverse saturation current (temperature dependent)
    - n: Ideality factor (quality of p-n junction)
    - R_s: Series resistance (metal contacts, interconnections)
    - R_sh: Shunt resistance (leakage current paths)
    
    The single-diode equation is:
    I = I_ph - I_0 * [exp((V + I*R_s)/V_t) - 1] - (V + I*R_s)/R_sh
    
    where V_t = (N_s * n * k * T) / q is the thermal voltage
    """
    
    def __init__(self):
        """
        Initialize PV module with Standard Test Conditions (STC) parameters
        STC: 1000 W/m² irradiance, 25°C cell temperature, AM 1.5 spectrum
        """
        # ===== Manufacturer Datasheet Parameters (STC) =====
        self.V_oc = 37.3        # Open circuit voltage (V) - maximum voltage at zero current
        self.I_sc = 8.51        # Short circuit current (A) - maximum current at zero voltage
        self.V_mp = 30.5        # Voltage at Maximum Power Point (V)
        self.I_mp = 7.84        # Current at Maximum Power Point (A)
        self.P_max = 239.2      # Maximum power rating (W) = V_mp * I_mp
        self.N_s = 60           # Number of cells connected in series
        
        # ===== Environmental Conditions =====
        self.T = 25 + 273.15    # Operating temperature (K) - converted from Celsius
        self.G = 1000           # Solar irradiance (W/m²) - amount of sunlight
        self.G_ref = 1000       # Reference irradiance at STC (W/m²)
        self.T_ref = 25 + 273.15  # Reference temperature at STC (K)
        
        # ===== Physical Constants =====
        self.k = 1.381e-23      # Boltzmann constant (J/K) - relates temperature to energy
        self.q = 1.602e-19      # Elementary electron charge (C)
        
        # Calculate the 5 parameters of the single-diode model
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """
        Calculate the 5 parameters of the single-diode model
        
        These parameters are extracted from datasheet values using analytical
        or numerical methods. The approach here uses typical values and 
        established relationships between parameters.
        """
        # ===== Ideality Factor (n) =====
        # Range: 1.0 (ideal diode) to 1.5 (non-ideal with recombination)
        # Higher values indicate more recombination losses in the p-n junction
        self.n = 1.3
        
        # ===== Thermal Voltage (V_t) =====
        # Represents the voltage equivalent of temperature
        # Higher temperature → higher thermal voltage → lower V_oc
        V_t = self.N_s * self.n * self.k * self.T / self.q
        
        # ===== Series Resistance (R_s) =====
        # Represents resistance in metal contacts, busbars, and interconnections
        # Lower R_s → better performance (steeper I-V curve slope)
        # Typical range: 0.1 - 1.0 Ω
        self.R_s = 0.39
        
        # ===== Shunt Resistance (R_sh) =====
        # Represents leakage current paths and manufacturing defects
        # Higher R_sh → better performance (more rectangular I-V curve)
        # Typical range: 100 - 1000 Ω
        self.R_sh = 294.0
        
        # ===== Photo-generated Current (I_ph) =====
        # Directly proportional to solar irradiance
        # More sunlight → more photons → more electron-hole pairs → higher I_ph
        self.I_ph = self.I_sc * (self.G / self.G_ref)
        
        # ===== Reverse Saturation Current (I_0) =====
        # Temperature-dependent parameter representing diode dark current
        # Calculated from the condition at open circuit (I = 0, V = V_oc)
        self.I_0 = (self.I_sc - (self.V_oc / self.R_sh)) / \
                   (np.exp(self.V_oc / V_t) - 1)
        
        # Display calculated parameters
        print("=" * 70)
        print("PV MODULE PARAMETERS (Single-Diode Model)")
        print("=" * 70)
        print(f"Photo-generated current (I_ph): {self.I_ph:.4f} A")
        print(f"Reverse saturation current (I_0): {self.I_0:.4e} A")
        print(f"Ideality factor (n): {self.n:.4f}")
        print(f"Series resistance (R_s): {self.R_s:.4f} Ω")
        print(f"Shunt resistance (R_sh): {self.R_sh:.4f} Ω")
        print(f"Thermal voltage (V_t): {V_t:.4f} V")
        print("=" * 70)
    
    def get_current(self, V):
        """
        Calculate current for given voltage using single-diode model
        
        The single-diode equation is implicit (I appears on both sides),
        so we use Newton-Raphson iterative method to solve it:
        
        I = I_ph - I_0 * [exp((V + I*R_s)/V_t) - 1] - (V + I*R_s)/R_sh
        
        Args:
            V: Voltage (V) - can be scalar or array
            
        Returns:
            I: Current (A) - same shape as input V
        """
        # Calculate thermal voltage for current temperature
        V_t = self.N_s * self.n * self.k * self.T / self.q
        
        # Ensure V is numpy array for vectorized operations
        V = np.atleast_1d(V)
        
        # Initialize current array
        I = np.zeros_like(V, dtype=float)
        
        # ===== Newton-Raphson Iterative Solution =====
        # For each voltage point, solve the implicit equation
        for idx, v in enumerate(V):
            # Initial guess: use I_ph as starting point (better than zero)
            I_old = self.I_ph
            
            # Iterate until convergence
            for iteration in range(1000):  # Maximum 1000 iterations
                
                # ===== Calculate Function Value f(I) =====
                # f(I) = I - I_ph + I_0*[exp((V+I*R_s)/V_t) - 1] + (V+I*R_s)/R_sh
                # We want f(I) = 0
                
                # Clip exponential argument to prevent overflow
                # (large positive values would cause exp() to overflow)
                exp_arg = np.clip((v + I_old * self.R_s) / V_t, -100, 100)
                
                # Function value
                f = I_old - self.I_ph + self.I_0 * (np.exp(exp_arg) - 1) + \
                    (v + I_old * self.R_s) / self.R_sh
                
                # ===== Calculate Derivative df/dI =====
                # df/dI = 1 + (I_0*R_s/V_t)*exp((V+I*R_s)/V_t) + R_s/R_sh
                df = 1 + (self.I_0 * self.R_s / V_t) * np.exp(exp_arg) + \
                     self.R_s / self.R_sh
                
                # ===== Newton-Raphson Update =====
                # I_new = I_old - f(I_old) / f'(I_old)
                I_new = I_old - f / df
                
                # ===== Check Convergence =====
                # If change is very small, we've found the solution
                if abs(I_new - I_old) < 1e-8:
                    break
                    
                I_old = I_new
            
            # Current cannot be negative (physical constraint)
            I[idx] = max(0, I_new)
        
        # Return scalar if input was scalar, array otherwise
        return I if len(I) > 1 else I[0]
    
    def get_power(self, V):
        """
        Calculate power output for given voltage
        
        Power = Voltage × Current
        
        Args:
            V: Voltage (V)
            
        Returns:
            P: Power (W)
        """
        I = self.get_current(V)
        return V * I
    
    def plot_characteristics(self):
        """
        Plot I-V and P-V characteristic curves of the PV module
        
        I-V Curve: Shows how current varies with voltage
        P-V Curve: Shows how power varies with voltage (has clear maximum)
        
        These curves are fundamental for understanding PV behavior and
        identifying the Maximum Power Point (MPP)
        """
        # Generate voltage points from 0 to V_oc
        V = np.linspace(0, self.V_oc, 500)
        
        # Calculate corresponding current and power
        I = self.get_current(V)
        P = V * I
        
        # ===== Find Maximum Power Point =====
        mpp_idx = np.argmax(P)  # Index where power is maximum
        V_mpp = V[mpp_idx]       # Voltage at MPP
        I_mpp = I[mpp_idx]       # Current at MPP
        P_mpp = P[mpp_idx]       # Maximum power
        
        # Display MPP information
        print("\nCALCULATED MAXIMUM POWER POINT:")
        print(f"Voltage at MPP (V_mpp): {V_mpp:.4f} V")
        print(f"Current at MPP (I_mpp): {I_mpp:.4f} A")
        print(f"Power at MPP (P_mpp): {P_mpp:.4f} W")
        print("=" * 70)
        
        # ===== Create Plots =====
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ----- I-V Characteristic Curve -----
        ax1.plot(V, I, 'b-', linewidth=2, label='I-V Curve')
        ax1.plot(V_mpp, I_mpp, 'ro', markersize=10, 
                label=f'MPP ({V_mpp:.2f}V, {I_mpp:.2f}A)')
        ax1.set_xlabel('Voltage (V)', fontsize=12)
        ax1.set_ylabel('Current (A)', fontsize=12)
        ax1.set_title('PV Module I-V Characteristic', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, self.V_oc])
        ax1.set_ylim([0, self.I_sc * 1.1])
        
        # ----- P-V Characteristic Curve -----
        ax2.plot(V, P, 'r-', linewidth=2, label='P-V Curve')
        ax2.plot(V_mpp, P_mpp, 'go', markersize=10, 
                label=f'MPP ({V_mpp:.2f}V, {P_mpp:.2f}W)')
        ax2.set_xlabel('Voltage (V)', fontsize=12)
        ax2.set_ylabel('Power (W)', fontsize=12)
        ax2.set_title('PV Module P-V Characteristic', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([0, self.V_oc])
        ax2.set_ylim([0, P_mpp * 1.1])
        
        plt.tight_layout()
        plt.savefig('pv_characteristics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return V_mpp, I_mpp, P_mpp


# =================================================================================
# BOOST CONVERTER MODEL
# =================================================================================

class BoostConverter:
    """
    DC-DC Boost Converter Model for PV System
    
    A boost converter steps up (increases) voltage from input to output.
    The relationship between input voltage, output voltage, and duty ratio is:
    
    V_out = V_in / (1 - D)
    
    Rearranging: V_in = V_out * (1 - D)
    
    Where:
    - D is the duty ratio (0 < D < 1)
    - D = T_on / T_period (fraction of time switch is ON)
    
    Key Components:
    - Inductor (L): Stores energy when switch is ON
    - Capacitor (C): Smooths output voltage
    - Diode: Allows current flow when switch is OFF
    - Switch: Controlled by PWM signal with duty ratio D
    """
    
    def __init__(self, pv_module):
        """
        Initialize boost converter with design parameters
        
        Args:
            pv_module: PVModule object representing the solar panel
        """
        self.pv = pv_module
        
        # ===== Converter Design Parameters =====
        self.V_out = 48.0       # Output voltage (V) - desired DC bus voltage
        self.L = 1e-3           # Inductance (H) = 1 mH - energy storage element
        self.C = 470e-6         # Capacitance (F) = 470 µF - voltage smoothing
        self.f_sw = 20000       # Switching frequency (Hz) = 20 kHz
        self.efficiency = 0.95  # Converter efficiency (95%) - accounts for losses
    
    def get_input_voltage(self, duty_ratio):
        """
        Calculate input voltage from duty ratio using boost converter equation
        
        From boost converter theory:
        V_out = V_in / (1 - D)
        
        Therefore:
        V_in = V_out * (1 - D)
        
        Args:
            duty_ratio (D): PWM duty cycle (0 to 1)
            
        Returns:
            V_in: Input voltage (V)
            
        Note: Duty ratio is clipped to (0.01, 0.99) to avoid:
        - D → 0: Very low input voltage
        - D → 1: Division by zero and infinite voltage
        """
        D = np.clip(duty_ratio, 0.01, 0.99)  # Ensure valid range
        return self.V_out * (1 - D)
    
    def get_output_power(self, duty_ratio):
        """
        Calculate output power considering converter efficiency
        
        P_out = P_in × efficiency
        
        Args:
            duty_ratio: PWM duty cycle
            
        Returns:
            P_out: Output power (W)
        """
        V_in = self.get_input_voltage(duty_ratio)
        I_in = self.pv.get_current(V_in)
        P_in = V_in * I_in
        P_out = P_in * self.efficiency
        return P_out
    
    def get_input_power(self, duty_ratio):
        """
        Calculate input power from PV module
        
        This is the power we want to maximize (MPPT objective)
        
        Args:
            duty_ratio: PWM duty cycle
            
        Returns:
            P_in: Input power (W) = V × I
        """
        V_in = self.get_input_voltage(duty_ratio)  # Get input voltage
        I_in = self.pv.get_current(V_in)           # Get current from PV at this voltage
        return V_in * I_in                          # Calculate power


# =================================================================================
# TASK 2: PERTURB AND OBSERVE (P&O) ALGORITHM
# =================================================================================

class PerturbObserve:
    """
    Perturb and Observe MPPT Algorithm
    
    This is the most widely used MPPT technique due to its simplicity.
    
    Algorithm Logic:
    1. Measure current power P(k)
    2. Perturb duty ratio: D(k+1) = D(k) ± ΔD
    3. Measure new power P(k+1)
    4. If P increased: continue in same direction
       If P decreased: reverse direction
    
    Advantages:
    - Simple to implement
    - No knowledge of PV characteristics needed
    - Works well in steady conditions
    
    Disadvantages:
    - Oscillates around MPP (never settles exactly)
    - Can be confused by rapidly changing irradiance
    - Trade-off between speed (large ΔD) and accuracy (small ΔD)
    """
    
    def __init__(self, boost_converter, delta_D=0.005, D_init=0.3):
        """
        Initialize P&O algorithm
        
        Args:
            boost_converter: BoostConverter object
            delta_D: Step size for perturbation (trade-off: speed vs accuracy)
            D_init: Initial duty ratio guess
        """
        self.boost = boost_converter
        self.delta_D = delta_D  # Perturbation step size
        self.D = D_init         # Current duty ratio
        
        # ===== History Arrays for Tracking =====
        self.D_history = []     # Duty ratio at each iteration
        self.P_history = []     # Power at each iteration
        self.V_history = []     # Voltage at each iteration
        self.I_history = []     # Current at each iteration
        
        # ===== Previous Values for Comparison =====
        self.P_prev = 0         # Power at previous step
        self.D_prev = D_init    # Duty ratio at previous step
    
    def step(self):
        """
        Execute one iteration of P&O algorithm
        
        This implements the core P&O logic:
        - Calculate power change (dP) and duty ratio change (dD)
        - Determine direction to perturb based on dP/dD slope
        """
        # ===== Measure Current Operating Point =====
        P_current = self.boost.get_input_power(self.D)
        V_current = self.boost.get_input_voltage(self.D)
        I_current = self.boost.pv.get_current(V_current)
        
        # ===== Store History =====
        self.D_history.append(self.D)
        self.P_history.append(P_current)
        self.V_history.append(V_current)
        self.I_history.append(I_current)
        
        # ===== P&O Decision Logic =====
        # Calculate changes from previous iteration
        dP = P_current - self.P_prev  # Change in power
        dD = self.D - self.D_prev      # Change in duty ratio
        
        # Only perturb if power change is significant (avoid noise)
        if abs(dP) > 0.01:
            # ===== Hill-Climbing Logic =====
            # We're trying to climb the P-V curve to reach the peak
            
            if dP > 0:  # Power increased
                if dD > 0:  # We moved right and power increased
                    self.D += self.delta_D  # Continue moving right
                else:       # We moved left and power increased
                    self.D -= self.delta_D  # Continue moving left
            else:       # Power decreased
                if dD > 0:  # We moved right and power decreased
                    self.D -= self.delta_D  # Reverse direction (go left)
                else:       # We moved left and power decreased
                    self.D += self.delta_D  # Reverse direction (go right)
        else:
            # Small perturbation to check if we're still at MPP
            # This helps escape from local optima
            self.D += self.delta_D
        
        # ===== Apply Physical Constraints =====
        # Duty ratio must be between 0.01 and 0.99
        self.D = np.clip(self.D, 0.01, 0.99)
        
        # ===== Update Previous Values for Next Iteration =====
        self.P_prev = P_current
        self.D_prev = self.D_history[-1]
    
    def run(self, iterations=100):
        """
        Run P&O algorithm for specified number of iterations
        
        Args:
            iterations: Number of P&O steps to execute
            
        Returns:
            Tuple of (optimal_D, max_P, optimal_V, optimal_I)
        """
        print("\n" + "=" * 70)
        print("TASK 2: PERTURB AND OBSERVE (P&O) ALGORITHM")
        print("=" * 70)
        
        # ===== Execute P&O Algorithm =====
        for i in range(iterations):
            self.step()
        
        # ===== Extract Results =====
        # Find the iteration with maximum power
        optimal_idx = np.argmax(self.P_history)
        optimal_D = self.D_history[optimal_idx]
        max_P = self.P_history[optimal_idx]
        optimal_V = self.V_history[optimal_idx]
        optimal_I = self.I_history[optimal_idx]
        
        # ===== Display Results =====
        print(f"\nP&O Algorithm Results:")
        print(f"Optimal Duty Ratio: {optimal_D:.6f}")
        print(f"Maximum Power: {max_P:.4f} W")
        print(f"Voltage at MPP: {optimal_V:.4f} V")
        print(f"Current at MPP: {optimal_I:.4f} A")
        print(f"Iterations: {iterations}")
        print(f"Final Duty Ratio: {self.D:.6f}")
        
        # Check if algorithm has converged (low oscillation)
        convergence_check = np.std(self.P_history[-10:]) < 0.5
        print(f"Convergence: {'Yes' if convergence_check else 'Oscillating'}")
        print("=" * 70)
        
        return optimal_D, max_P, optimal_V, optimal_I
    
    def plot_results(self):
        """
        Visualize P&O algorithm performance
        
        Plots show:
        1. Power tracking over iterations
        2. Duty ratio variation (shows oscillation around optimum)
        3. Voltage tracking
        4. Current tracking
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(self.P_history))
        
        # ----- Power vs Iteration -----
        # Shows how power converges to maximum
        ax1.plot(iterations, self.P_history, 'b-', linewidth=1.5)
        ax1.axhline(y=max(self.P_history), color='r', linestyle='--', 
                    label=f'Max Power = {max(self.P_history):.2f} W')
        ax1.set_xlabel('Iteration', fontsize=11)
        ax1.set_ylabel('Power (W)', fontsize=11)
        ax1.set_title('Power Tracking - P&O Algorithm', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # ----- Duty Ratio vs Iteration -----
        # Shows oscillation behavior around optimal value
        ax2.plot(iterations, self.D_history, 'g-', linewidth=1.5)
        optimal_D = self.D_history[np.argmax(self.P_history)]
        ax2.axhline(y=optimal_D, color='r', linestyle='--', 
                    label=f'Optimal D = {optimal_D:.4f}')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Duty Ratio', fontsize=11)
        ax2.set_title('Duty Ratio Variation - P&O Algorithm', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # ----- Voltage vs Iteration -----
        ax3.plot(iterations, self.V_history, 'm-', linewidth=1.5)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Voltage (V)', fontsize=11)
        ax3.set_title('Voltage Tracking - P&O Algorithm', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ----- Current vs Iteration -----
        ax4.plot(iterations, self.I_history, 'c-', linewidth=1.5)
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Current (A)', fontsize=11)
        ax4.set_title('Current Tracking - P&O Algorithm', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('po_algorithm_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# =================================================================================
# TASK 3: NON-LINEAR OPTIMIZATION
# =================================================================================

class NonLinearOptimization:
    """
    Non-Linear Optimization for MPPT using Gradient-Based Methods
    
    This approach treats MPPT as a mathematical optimization problem:
    
    maximize: P(D) = V(D) × I(D)
    subject to: 0.01 ≤ D ≤ 0.99
    
    We use scipy.optimize.minimize with two algorithms:
    1. SLSQP (Sequential Least Squares Programming)
       - Handles constraints efficiently
       - Uses gradient information
       - Fast convergence
    
    2. L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
       - Quasi-Newton method
       - Approximates Hessian matrix
       - Memory efficient
    
    Advantages:
    - Fast convergence (few iterations)
    - High accuracy
    - Mathematical guarantee of optimality (for convex problems)
    
    Disadvantages:
    - Requires differentiable objective function
    - May get stuck in local optima (if multiple peaks exist)
    - Computationally more intensive per iteration
    """
    
    def __init__(self, boost_converter):
        """
        Initialize non-linear optimizer
        
        Args:
            boost_converter: BoostConverter object
        """
        self.boost = boost_converter
    
    def objective(self, D):
        """
        Objective function to MINIMIZE (negative power)
        
        scipy.optimize.minimize finds minimum, but we want maximum power.
        Solution: minimize negative power = maximize power
        
        Args:
            D: Duty ratio (as array for scipy interface)
            
        Returns:
            -P: Negative of power (to convert maximization to minimization)
        """
        return -self.boost.get_input_power(D[0])
    
    def optimize(self):
        """
        Perform non-linear optimization using two different algorithms
        
        Returns:
            Tuple of (result_slsqp, result_lbfgsb) containing optimization results
        """
        print("\n" + "=" * 70)
        print("TASK 3: NON-LINEAR OPTIMIZATION")
        print("=" * 70)
        
        # ===== Problem Setup =====
        D0 = [0.3]  # Initial guess for duty ratio
        bounds = [(0.01, 0.99)]  # Constraints on duty ratio
        
        # ========== Method 1: SLSQP ==========
        print("\nUsing SLSQP (Sequential Least Squares Programming)...")
        
        # SLSQP is good for:
        # - Constrained optimization
        # - Smooth objective functions
        # - When gradients are available (or can be approximated)
        result_slsqp = minimize(
            self.objective,      # Function to minimize
            D0,                  # Initial guess
            method='SLSQP',      # Algorithm choice
            bounds=bounds,       # Variable bounds
            options={
                'ftol': 1e-9,    # Function tolerance (stop when change < ftol)
                'maxiter': 1000  # Maximum iterations
            }
        )
        
        # Extract results
        optimal_D_slsqp = result_slsqp.x[0]
        max_P_slsqp = -result_slsqp.fun  # Convert back to positive (we minimized negative)
        optimal_V_slsqp = self.boost.get_input_voltage(optimal_D_slsqp)
        optimal_I_slsqp = self.boost.pv.get_current(optimal_V_slsqp)
        
        # Display SLSQP results
        print(f"\nSLSQP Results:")
        print(f"Optimal Duty Ratio: {optimal_D_slsqp:.6f}")
        print(f"Maximum Power: {max_P_slsqp:.4f} W")
        print(f"Voltage at MPP: {optimal_V_slsqp:.4f} V")
        print(f"Current at MPP: {optimal_I_slsqp:.4f} A")
        print(f"Success: {result_slsqp.success}")
        print(f"Function Evaluations: {result_slsqp.nfev}")
        
        # ========== Method 2: L-BFGS-B ==========
        print("\nUsing L-BFGS-B (Limited-memory BFGS)...")
        
        # L-BFGS-B is good for:
        # - Large-scale optimization (memory efficient)
        # - Smooth functions
        # - Box constraints (simple bounds)
        result_lbfgsb = minimize(
            self.objective,
            D0,
            method='L-BFGS-B',  # Quasi-Newton method
            bounds=bounds,
            options={
                'ftol': 1e-9,
                'maxiter': 1000
            }
        )
        
        # Extract results
        optimal_D_lbfgsb = result_lbfgsb.x[0]
        max_P_lbfgsb = -result_lbfgsb.fun
        optimal_V_lbfgsb = self.boost.get_input_voltage(optimal_D_lbfgsb)
        optimal_I_lbfgsb = self.boost.pv.get_current(optimal_V_lbfgsb)
        
        # Display L-BFGS-B results
        print(f"\nL-BFGS-B Results:")
        print(f"Optimal Duty Ratio: {optimal_D_lbfgsb:.6f}")
        print(f"Maximum Power: {max_P_lbfgsb:.4f} W")
        print(f"Voltage at MPP: {optimal_V_lbfgsb:.4f} V")
        print(f"Current at MPP: {optimal_I_lbfgsb:.4f} A")
        print(f"Success: {result_lbfgsb.success}")
        print(f"Function Evaluations: {result_lbfgsb.nfev}")
        
        print("=" * 70)
        
        return result_slsqp, result_lbfgsb
    
    def plot_landscape(self):
        """
        Visualize the power landscape (objective function)
        
        This shows how power varies with duty ratio, helping understand:
        - Whether the problem is convex (single peak)
        - Where the global optimum is located
        - The shape of the objective function
        """
        # Generate duty ratio range
        D_range = np.linspace(0.01, 0.99, 200)
        
        # Calculate power for each duty ratio
        P_range = [self.boost.get_input_power(d) for d in D_range]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(D_range, P_range, 'b-', linewidth=2, label='Power Landscape')
        
        # Mark optimal point
        optimal_idx = np.argmax(P_range)
        ax.plot(D_range[optimal_idx], P_range[optimal_idx], 'ro', 
                markersize=12, label=f'Optimal Point (D={D_range[optimal_idx]:.4f})')
        
        ax.set_xlabel('Duty Ratio', fontsize=12)
        ax.set_ylabel('Power (W)', fontsize=12)
        ax.set_title('Power vs Duty Ratio - Non-Linear Optimization', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig('nonlinear_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()


# =================================================================================
# TASK 4: PARTICLE SWARM OPTIMIZATION (PSO)
# =================================================================================

class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for MPPT
    
    PSO is a population-based metaheuristic inspired by bird flocking behavior.
    
    Concept:
    - Multiple "particles" explore the search space
    - Each particle has position (duty ratio) and velocity
    - Particles remember their best position (personal best)
    - Swarm shares global best position
    - Particles balance exploration (randomness) and exploitation (convergence)
    
    Update Equations:
    v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
    x(t+1) = x(t) + v(t+1)
    
    Where:
    - w: Inertia weight (momentum, prevents premature convergence)
    - c1: Cognitive parameter (attraction to personal best)
    - c2: Social parameter (attraction to global best)
    - r1, r2: Random numbers [0,1] (adds stochastic exploration)
    
    Advantages:
    - Global optimization (can find global optimum even with multiple peaks)
    - Derivative-free (works with non-differentiable functions)
    - Robust to noise and uncertainties
    - Parallelizable (particles can be evaluated independently)
    
    Disadvantages:
    - More function evaluations than gradient methods
    - Performance depends on parameter tuning
    - No mathematical convergence guarantee
    """
    
    def __init__(self, boost_converter, n_particles=30, max_iter=50):
        """
        Initialize PSO algorithm
        
        Args:
            boost_converter: BoostConverter object
            n_particles: Number of particles in swarm (trade-off: diversity vs computation)
            max_iter: Maximum iterations (stopping criterion)
        """
        self.boost = boost_converter
        self.n_particles = n_particles
        self.max_iter = max_iter
        
        # ===== PSO Parameters (Constriction Coefficient Approach) =====
        # These values are from Clerc and Kennedy's constriction coefficient
        self.w = 0.7298      # Inertia weight (balances exploration/exploitation)
        self.c1 = 1.49618    # Cognitive parameter (personal learning)
        self.c2 = 1.49618    # Social parameter (swarm learning)
        
        # ===== Search Space Bounds =====
        self.D_min = 0.01    # Minimum duty ratio
        self.D_max = 0.99    # Maximum duty ratio
        
        # ===== Initialize Particle Swarm =====
        np.random.seed(42)
        
        # Initialize positions randomly across search space
        self.positions = np.random.uniform(self.D_min, self.D_max, self.n_particles)
        
        # Initialize velocities with small random values
        self.velocities = np.random.uniform(-0.1, 0.1, self.n_particles)
        
        # ===== Personal Best Tracking =====
        # Each particle remembers its best position found so far
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.array([self.fitness(p) for p in self.positions])
        
        # ===== Global Best Tracking =====
        # Swarm shares the best position found by any particle
        best_idx = np.argmax(self.pbest_scores)
        self.gbest_position = self.pbest_positions[best_idx]
        self.gbest_score = self.pbest_scores[best_idx]
        
        # ===== History for Visualization =====
        self.gbest_history = []        # Global best over iterations
        self.mean_fitness_history = []  # Mean fitness over iterations
    
    def fitness(self, D):
        """
        Fitness function (power to be maximized)
        
        Higher fitness = better solution
        
        Args:
            D: Duty ratio
            
        Returns:
            Power at this duty ratio
        """
        D_clipped = np.clip(D, self.D_min, self.D_max)
        return self.boost.get_input_power(D_clipped)
    
    def optimize(self):
        """
        Run PSO optimization algorithm
        
        Returns:
            Tuple of (optimal_D, max_P, optimal_V, optimal_I)
        """
        print("\n" + "=" * 70)
        print("TASK 4: PARTICLE SWARM OPTIMIZATION (PSO)")
        print("=" * 70)
        print(f"Number of particles: {self.n_particles}")
        print(f"Maximum iterations: {self.max_iter}")
        print(f"Inertia weight (w): {self.w}")
        print(f"Cognitive parameter (c1): {self.c1}")
        print(f"Social parameter (c2): {self.c2}\n")
        
        # ===== PSO Main Loop =====
        for iteration in range(self.max_iter):
            
            # Update each particle
            for i in range(self.n_particles):
                
                # ===== Generate Random Factors =====
                # These add stochastic element to the search
                r1, r2 = np.random.rand(), np.random.rand()
                
                # ===== Calculate Velocity Components =====
                
                # Cognitive component: attraction to personal best
                # "I should go back to where I did well"
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                
                # Social component: attraction to global best
                # "I should go where the swarm did well"
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])
                
                # ===== Update Velocity =====
                # v(t+1) = w*v(t) + cognitive + social
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # ===== Velocity Clamping =====
                # Prevent particles from moving too fast (maintains diversity)
                v_max = 0.2 * (self.D_max - self.D_min)
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
                
                # ===== Update Position =====
                # x(t+1) = x(t) + v(t+1)
                self.positions[i] += self.velocities[i]
                
                # ===== Apply Boundary Constraints =====
                # Keep particles within valid duty ratio range
                self.positions[i] = np.clip(self.positions[i], self.D_min, self.D_max)
                
                # ===== Evaluate Fitness =====
                fitness = self.fitness(self.positions[i])
                
                # ===== Update Personal Best =====
                if fitness > self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i]
                
                # ===== Update Global Best =====
                if fitness > self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i]
            
            # ===== Store History =====
            self.gbest_history.append(self.gbest_score)
            self.mean_fitness_history.append(np.mean(self.pbest_scores))
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Best Power = {self.gbest_score:.4f} W")
        
        # ===== Extract Final Results =====
        optimal_D = self.gbest_position
        max_P = self.gbest_score
        optimal_V = self.boost.get_input_voltage(optimal_D)
        optimal_I = self.boost.pv.get_current(optimal_V)
        
        print(f"\nPSO Results:")
        print(f"Optimal Duty Ratio: {optimal_D:.6f}")
        print(f"Maximum Power: {max_P:.4f} W")
        print(f"Voltage at MPP: {optimal_V:.4f} V")
        print(f"Current at MPP: {optimal_I:.4f} A")
        print("=" * 70)
        
        return optimal_D, max_P, optimal_V, optimal_I
    
    def plot_results(self):
        """
        Visualize PSO optimization process
        
        Shows:
        1. Convergence of global best and mean fitness
        2. Final particle distribution on power landscape
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        iterations = range(1, len(self.gbest_history) + 1)
        
        # ----- Convergence Plot -----
        # Shows how swarm converges to optimum over iterations
        ax1.plot(iterations, self.gbest_history, 'b-', linewidth=2, 
                 label='Global Best')
        ax1.plot(iterations, self.mean_fitness_history, 'r--', linewidth=2, 
                 label='Mean Fitness')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Power (W)', fontsize=12)
        ax1.set_title('PSO Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # ----- Particle Distribution -----
        # Shows final particle positions on power landscape
        D_range = np.linspace(self.D_min, self.D_max, 200)
        P_range = [self.fitness(d) for d in D_range]
        
        ax2.plot(D_range, P_range, 'k-', linewidth=2, alpha=0.5, label='Power Landscape')
        ax2.scatter(self.positions, [self.fitness(p) for p in self.positions], 
                   c='blue', s=50, alpha=0.6, label='Final Particles')
        ax2.scatter(self.gbest_position, self.gbest_score, 
                   c='red', s=200, marker='*', label='Global Best', zorder=5)
        ax2.set_xlabel('Duty Ratio', fontsize=12)
        ax2.set_ylabel('Power (W)', fontsize=12)
        ax2.set_title('PSO Particle Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('pso_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# =================================================================================
# TASK 5: ARTIFICIAL NEURAL NETWORK (ANN)
# =================================================================================

class ANNOptimization:
    """
    Artificial Neural Network for MPPT
    
    Unlike optimization algorithms, ANN learns the mapping:
    (Temperature, Irradiance, Voltage) → Optimal Duty Ratio
    
    Concept:
    1. Generate training data from PV model under various conditions
    2. Train neural network to learn the relationship
    3. Use trained network for real-time MPPT (very fast inference)
    
    Network Architecture:
    Input Layer: 3 neurons (Temperature, Irradiance, Voltage)
         ↓
    Hidden Layer 1: 10 neurons (ReLU activation)
         ↓
    Hidden Layer 2: 8 neurons (ReLU activation)
         ↓
    Output Layer: 1 neuron (Sigmoid activation → duty ratio ∈ [0,1])
    
    Advantages:
    - Very fast inference (once trained)
    - Adapts to varying environmental conditions
    - Can learn complex non-linear relationships
    - Robust to noise in measurements
    
    Disadvantages:
    - Requires extensive training data
    - Training is computationally intensive
    - Needs periodic retraining for aging panels
    - "Black box" - hard to interpret decisions
    """
    
    def __init__(self, boost_converter):
        """
        Initialize ANN for MPPT
        
        Args:
            boost_converter: BoostConverter object
        """
        self.boost = boost_converter
        self.pv = boost_converter.pv
        
        # ===== Network Architecture =====
        self.input_size = 3      # T, G, V
        self.hidden_size1 = 10   # First hidden layer
        self.hidden_size2 = 8    # Second hidden layer
        self.output_size = 1     # Duty ratio
        
        # Initialize network weights
        self._initialize_weights()
        
        # ===== Data Storage =====
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # ===== Normalization Parameters =====
        # Stored for consistent scaling during prediction
        self.T_mean = None
        self.T_std = None
        self.G_mean = None
        self.G_std = None
        self.V_mean = None
        self.V_std = None
        
        # ===== Training History =====
        self.loss_history = []
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization
        
        He initialization is optimal for ReLU activation:
        W ~ N(0, sqrt(2/n_in))
        
        This prevents:
        - Vanishing gradients (weights too small)
        - Exploding gradients (weights too large)
        """
        np.random.seed(42)
        
        # ===== Layer 1: Input → Hidden1 =====
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * \
                  np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size1))
        
        # ===== Layer 2: Hidden1 → Hidden2 =====
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * \
                  np.sqrt(2.0 / self.hidden_size1)
        self.b2 = np.zeros((1, self.hidden_size2))
        
        # ===== Layer 3: Hidden2 → Output =====
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * \
                  np.sqrt(2.0 / self.hidden_size2)
        self.b3 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        """
        Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
        
        Properties:
        - Output range: (0, 1)
        - Perfect for duty ratio (must be between 0 and 1)
        - Smooth and differentiable
        
        Args:
            x: Input values
            
        Returns:
            Sigmoid of x
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        
        Used in backpropagation
        """
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        """
        ReLU activation function: f(x) = max(0, x)
        
        Properties:
        - Introduces non-linearity
        - Computationally efficient
        - Helps with vanishing gradient problem
        - Used in hidden layers
        
        Args:
            x: Input values
            
        Returns:
            ReLU of x
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of ReLU: f'(x) = 1 if x > 0, else 0
        
        Used in backpropagation
        """
        return (x > 0).astype(float)
    
    def forward(self, X):
        """
        Forward propagation through the network
        
        Computes output for given inputs by passing through all layers
        
        Process:
        Input → [W1, b1] → ReLU → [W2, b2] → ReLU → [W3, b3] → Sigmoid → Output
        
        Args:
            X: Input matrix (batch_size × 3)
            
        Returns:
            Output predictions (batch_size × 1)
        """
        # ===== Layer 1 =====
        # Linear transformation: z = W·x + b
        self.z1 = np.dot(X, self.W1) + self.b1
        # Activation: a = ReLU(z)
        self.a1 = self.relu(self.z1)
        
        # ===== Layer 2 =====
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # ===== Output Layer =====
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        # Sigmoid ensures output ∈ (0, 1)
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Backpropagation algorithm for training
        
        Computes gradients and updates weights to minimize error
        Uses chain rule to propagate error backward through network
        
        Loss Function: MSE = (1/m) * Σ(predicted - actual)²
        
        Args:
            X: Input data
            y: True labels (target duty ratios)
            learning_rate: Step size for weight updates
        """
        m = X.shape[0]  # Number of samples
        
        # ===== Output Layer Gradients =====
        # Error at output: dL/dz3 = (predicted - actual)
        dz3 = self.a3 - y
        
        # Gradient w.r.t. weights: dL/dW3 = (1/m) * a2^T · dz3
        dW3 = (1/m) * np.dot(self.a2.T, dz3)
        
        # Gradient w.r.t. bias: dL/db3 = (1/m) * Σdz3
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # ===== Hidden Layer 2 Gradients =====
        # Backpropagate error: dL/da2 = dz3 · W3^T
        da2 = np.dot(dz3, self.W3.T)
        
        # Apply activation derivative: dL/dz2 = da2 ⊙ ReLU'(z2)
        dz2 = da2 * self.relu_derivative(self.z2)
        
        # Gradients for W2 and b2
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # ===== Hidden Layer 1 Gradients =====
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # ===== Gradient Clipping =====
        # Prevent exploding gradients
        max_grad = 5.0
        dW3 = np.clip(dW3, -max_grad, max_grad)
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW1 = np.clip(dW1, -max_grad, max_grad)
        
        # ===== Weight Updates (Gradient Descent) =====
        # W_new = W_old - learning_rate * gradient
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate synthetic training data from PV model
        
        Creates diverse dataset by varying:
        - Temperature: 15°C to 45°C (typical operating range)
        - Irradiance: 200 to 1200 W/m² (cloudy to bright sun)
        - Voltage: 5V to 40V (operating range)
        
        For each condition, calculates optimal duty ratio
        
        Args:
            n_samples: Number of training samples to generate
        """
        print("\n" + "=" * 70)
        print("TASK 5: ARTIFICIAL NEURAL NETWORK (ANN)")
        print("=" * 70)
        print(f"Generating {n_samples} training samples...")
        
        np.random.seed(42)
        
        # ===== Generate Random Operating Conditions =====
        temperatures = np.random.uniform(15, 45, n_samples) + 273.15  # K
        irradiances = np.random.uniform(200, 1200, n_samples)  # W/m²
        voltages = np.random.uniform(5, 40, n_samples)  # V
        
        # ===== Calculate Optimal Duty Ratios =====
        duty_ratios = []
        
        # Store original conditions
        T_orig = self.pv.T
        G_orig = self.pv.G
        
        for i in range(n_samples):
            # Set PV to specific conditions
            self.pv.T = temperatures[i]
            self.pv.G = irradiances[i]
            self.pv._calculate_parameters()
            
            # Calculate duty ratio from boost converter equation
            # V_in = V_out * (1 - D)
            # D = 1 - V_in/V_out
            V = voltages[i]
            D = 1 - (V / self.boost.V_out)
            D = np.clip(D, 0.01, 0.99)  # Ensure valid range
            
            duty_ratios.append(D)
        
        # Restore original conditions
        self.pv.T = T_orig
        self.pv.G = G_orig
        self.pv._calculate_parameters()
        
        # ===== Normalize Inputs =====
        # Normalization: x_norm = (x - mean) / std
        # Benefits:
        # - Faster convergence
        # - Prevents any input from dominating
        # - Improves numerical stability
        
        self.T_mean, self.T_std = np.mean(temperatures), np.std(temperatures) + 1e-8
        self.G_mean, self.G_std = np.mean(irradiances), np.std(irradiances) + 1e-8
        self.V_mean, self.V_std = np.mean(voltages), np.std(voltages) + 1e-8
        
        T_norm = (temperatures - self.T_mean) / self.T_std
        G_norm = (irradiances - self.G_mean) / self.G_std
        V_norm = (voltages - self.V_mean) / self.V_std
        
        # ===== Create Dataset =====
        X = np.column_stack([T_norm, G_norm, V_norm])
        y = np.array(duty_ratios).reshape(-1, 1)
        
        # ===== Train/Test Split (80/20) =====
        split_idx = int(0.8 * n_samples)
        indices = np.random.permutation(n_samples)
        
        self.X_train = X[indices[:split_idx]]
        self.y_train = y[indices[:split_idx]]
        self.X_test = X[indices[split_idx:]]
        self.y_test = y[indices[split_idx:]]
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
    
    def train(self, epochs=500, learning_rate=0.01, batch_size=32):
        """
        Train the neural network using mini-batch gradient descent
        
        Process:
        1. Shuffle training data
        2. Divide into mini-batches
        3. For each batch:
           - Forward propagation
           - Calculate loss
           - Backward propagation
           - Update weights
        4. Repeat for all epochs
        
        Args:
            epochs: Number of complete passes through dataset
            learning_rate: Step size for weight updates
            batch_size: Number of samples per mini-batch
        """
        print(f"\nTraining ANN for {epochs} epochs...")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}\n")
        
        n_batches = max(1, len(self.X_train) // batch_size)
        
        # ===== Training Loop =====
        for epoch in range(epochs):
            # Shuffle training data each epoch (improves generalization)
            indices = np.random.permutation(len(self.X_train))
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            epoch_loss = 0
            
            # ===== Mini-Batch Training =====
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(self.X_train))
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Backward pass (updates weights)
                self.backward(X_batch, y_batch, learning_rate)
                
                # Calculate Mean Squared Error
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss
            
            # Average loss for this epoch
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # ===== Periodic Evaluation =====
            if (epoch + 1) % 100 == 0:
                # Test on validation set
                test_pred = self.forward(self.X_test)
                test_loss = np.mean((test_pred - self.y_test) ** 2)
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}")
        
        print("\nTraining completed!")
    
    def predict(self, T, G, V):
        """
        Predict optimal duty ratio for given conditions
        
        This is used in real-time after training
        
        Args:
            T: Temperature (K)
            G: Irradiance (W/m²)
            V: Voltage (V)
            
        Returns:
            Predicted duty ratio
        """
        # Normalize inputs using training statistics
        T_norm = (T - self.T_mean) / self.T_std
        G_norm = (G - self.G_mean) / self.G_std
        V_norm = (V - self.V_mean) / self.V_std
        
        # Create input array
        X = np.array([[T_norm, G_norm, V_norm]])
        
        # Forward propagation
        D_pred = self.forward(X)
        
        return float(D_pred[0, 0])
    
    def evaluate(self):
        """
        Evaluate trained network performance
        
        Calculates:
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - R²: Coefficient of determination (1.0 = perfect)
        """
        # Make predictions on test set
        test_predictions = self.forward(self.X_test)
        
        # ===== Calculate Metrics =====
        mse = np.mean((test_predictions - self.y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - self.y_test))
        
        # R² score: 1.0 = perfect, 0.0 = baseline, <0 = worse than baseline
        ss_res = np.sum((self.y_test - test_predictions) ** 2)  # Residual sum of squares
        ss_tot = np.sum((self.y_test - np.mean(self.y_test)) ** 2)  # Total sum of squares
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\nANN Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R² Score: {r2_score:.6f}")
        
        # ===== Test on Current Conditions =====
        T_current = self.pv.T
        G_current = self.pv.G
        V_current = self.pv.V_mp
        
        D_predicted = self.predict(T_current, G_current, V_current)
        P_predicted = self.boost.get_input_power(D_predicted)
        
        print(f"\nPrediction at Current Conditions:")
        print(f"Temperature: {T_current - 273.15:.2f}°C")
        print(f"Irradiance: {G_current:.2f} W/m²")
        print(f"Voltage: {V_current:.2f} V")
        print(f"Predicted Duty Ratio: {D_predicted:.6f}")
        print(f"Predicted Power: {P_predicted:.4f} W")
        print("=" * 70)
        
        return mse, rmse, mae, r2_score
    
    def plot_results(self):
        """
        Visualize ANN training and prediction results
        """
        fig = plt.figure(figsize=(15, 10))
        
        # ----- Training Loss (Log Scale) -----
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.loss_history, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('ANN Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale shows convergence better
        
        # ----- Predictions vs Actual -----
        ax2 = plt.subplot(2, 2, 2)
        test_predictions = self.forward(self.X_test)
        ax2.scatter(self.y_test, test_predictions, alpha=0.5, s=20)
        
        # Perfect prediction line (y = x)
        min_val = min(np.min(self.y_test), np.min(test_predictions))
        max_val = max(np.max(self.y_test), np.max(test_predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Duty Ratio', fontsize=11)
        ax2.set_ylabel('Predicted Duty Ratio', fontsize=11)
        ax2.set_title('ANN Predictions vs Actual', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # ----- Residual Plot -----
        # Should be random if model is good
        ax3 = plt.subplot(2, 2, 3)
        residuals = self.y_test - test_predictions
        ax3.scatter(test_predictions, residuals, alpha=0.5, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Duty Ratio', fontsize=11)
        ax3.set_ylabel('Residuals', fontsize=11)
        ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ----- Error Distribution -----
        # Should be approximately normal if model is good
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Prediction Error', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ann_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# =================================================================================
# TASK 6: DYNAMIC PROGRAMMING OPTIMIZATION
# =================================================================================

class DynamicProgramming:
    """
    Dynamic Programming for MPPT using Value Iteration
    
    Dynamic Programming solves sequential decision problems by:
    1. Breaking problem into states and actions
    2. Computing value of each state
    3. Finding optimal policy (action for each state)
    
    In MPPT Context:
    - States: Discretized duty ratio values
    - Actions: Small changes to duty ratio (perturbations)
    - Reward: Power output at current state
    - Value: Long-term power considering future states
    
    Bellman Equation (core of DP):
    V(s) = max_a [R(s, a) + γ * V(s')]
    
    Where:
    - V(s): Value of state s
    - R(s, a): Immediate reward for action a in state s
    - γ: Discount factor (0 < γ < 1)
    - s': Next state after action a
    
    Value Iteration Algorithm:
    1. Initialize V(s) = 0 for all states
    2. Repeat until convergence:
       For each state s:
         V_new(s) = max_a [R(s,a) + γ*V(s')]
    3. Extract optimal policy: π*(s) = argmax_a [R(s,a) + γ*V(s')]
    
    Advantages:
    - Guaranteed global optimum (for discretized space)
    - Systematic exploration of all possibilities
    - Can handle complex constraints
    - Policy can be reused
    
    Disadvantages:
    - Computational cost grows with state space size
    - Requires discretization (loses some precision)
    - Not suitable for high-dimensional problems
    """
    
    def __init__(self, boost_converter, n_states=100):
        """
        Initialize Dynamic Programming optimizer
        
        Args:
            boost_converter: BoostConverter object
            n_states: Number of discrete states (trade-off: precision vs computation)
        """
        self.boost = boost_converter
        self.n_states = n_states
        
        # ===== State Space Discretization =====
        # Divide continuous duty ratio range into discrete states
        self.D_min = 0.01
        self.D_max = 0.99
        self.states = np.linspace(self.D_min, self.D_max, n_states)
        
        # ===== Value Function and Policy =====
        # V[i] = value of being in state i
        # policy[i] = optimal action index to take from state i
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
        
        # ===== DP Parameters =====
        self.gamma = 0.95  # Discount factor (values future rewards at 95% of immediate)
        self.epsilon = 1e-6  # Convergence threshold
        
        # ===== Action Space =====
        # Actions are small perturbations to duty ratio
        self.delta_actions = [-0.05, -0.02, 0.0, 0.02, 0.05]
        self.n_actions = len(self.delta_actions)
        
        # ===== History for Visualization =====
        self.value_history = []  # Tracks convergence
        self.iteration_count = 0
    
    def reward_function(self, state_idx):
        """
        Reward function: immediate power at given state
        
        Higher power = higher reward
        
        Args:
            state_idx: Index of state in discretized space
            
        Returns:
            Power (reward) at this state
        """
        D = self.states[state_idx]
        power = self.boost.get_input_power(D)
        return power
    
    def get_next_state(self, state_idx, action_idx):
        """
        Transition function: returns next state given current state and action
        
        Implements state dynamics:
        D_next = D_current + Δ_action
        
        Args:
            state_idx: Current state index
            action_idx: Action index (perturbation)
            
        Returns:
            next_state_idx: Index of resulting state
        """
        # Get current duty ratio
        current_D = self.states[state_idx]
        
        # Apply action (perturbation)
        delta_D = self.delta_actions[action_idx]
        next_D = current_D + delta_D
        
        # Clip to valid range
        next_D = np.clip(next_D, self.D_min, self.D_max)
        
        # Find closest discrete state
        next_state_idx = np.argmin(np.abs(self.states - next_D))
        
        return next_state_idx
    
    def value_iteration(self, max_iterations=1000):
        """
        Value Iteration algorithm implementation
        
        Iteratively updates value function until convergence
        
        Algorithm:
        1. For each state s:
           a. Try all possible actions
           b. Calculate Q(s,a) = R(s,a) + γ*V(s')
           c. Set V(s) = max_a Q(s,a)
           d. Set policy(s) = argmax_a Q(s,a)
        2. Check convergence: if max change < ε, stop
        3. Otherwise, repeat
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (optimal_D, max_P, optimal_V, optimal_I)
        """
        print("\n" + "=" * 70)
        print("TASK 6: DYNAMIC PROGRAMMING (Value Iteration)")
        print("=" * 70)
        print(f"Number of states: {self.n_states}")
        print(f"Number of actions: {self.n_actions}")
        print(f"Discount factor (gamma): {self.gamma}")
        print(f"Convergence threshold: {self.epsilon}\n")
        
        print("Running Value Iteration...")
        
        # ===== Value Iteration Main Loop =====
        for iteration in range(max_iterations):
            delta = 0  # Maximum change in value function
            V_new = np.zeros(self.n_states)  # New value function
            
            # ===== Update Each State =====
            for s in range(self.n_states):
                v_old = self.V[s]  # Current value
                
                # ===== Try All Actions =====
                action_values = np.zeros(self.n_actions)
                
                for a in range(self.n_actions):
                    # Get next state from this action
                    s_next = self.get_next_state(s, a)
                    
                    # Calculate immediate reward
                    reward = self.reward_function(s)
                    
                    # ===== Bellman Equation =====
                    # Q(s,a) = R(s) + γ * V(s')
                    # Immediate reward + discounted future value
                    action_values[a] = reward + self.gamma * self.V[s_next]
                
                # ===== Update Value Function =====
                # V(s) = max_a Q(s,a)
                V_new[s] = np.max(action_values)
                
                # ===== Update Policy =====
                # π(s) = argmax_a Q(s,a)
                self.policy[s] = np.argmax(action_values)
                
                # Track maximum change for convergence check
                delta = max(delta, abs(V_new[s] - v_old))
            
            # ===== Update Value Function =====
            self.V = V_new.copy()
            self.value_history.append(delta)
            self.iteration_count = iteration + 1
            
            # Print progress
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: Max Value Change = {delta:.6f}")
            
            # ===== Check Convergence =====
            # If value function stopped changing significantly, we're done
            if delta < self.epsilon:
                print(f"\nConverged after {iteration + 1} iterations!")
                break
        
        # ===== Extract Optimal Solution =====
        # The state with highest value is the optimal duty ratio
        optimal_state_idx = np.argmax(self.V)
        optimal_D = self.states[optimal_state_idx]
        optimal_power = self.reward_function(optimal_state_idx)
        optimal_V = self.boost.get_input_voltage(optimal_D)
        optimal_I = self.boost.pv.get_current(optimal_V)
        
        print(f"\nDynamic Programming Results:")
        print(f"Optimal State Index: {optimal_state_idx}")
        print(f"Optimal Duty Ratio: {optimal_D:.6f}")
        print(f"Maximum Power: {optimal_power:.4f} W")
        print(f"Voltage at MPP: {optimal_V:.4f} V")
        print(f"Current at MPP: {optimal_I:.4f} A")
        print(f"Optimal Value: {self.V[optimal_state_idx]:.4f}")
        print(f"Total Iterations: {self.iteration_count}")
        print("=" * 70)
        
        return optimal_D, optimal_power, optimal_V, optimal_I
    
    def plot_results(self):
        """
        Visualize Dynamic Programming results
        
        Shows:
        1. Value function across all states
        2. Optimal policy (action) for each state
        3. Convergence over iterations
        4. Optimal point on power landscape
        """
        fig = plt.figure(figsize=(15, 10))
        
        # ----- Value Function -----
        # Shows value of each duty ratio state
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.states, self.V, 'b-', linewidth=2)
        optimal_idx = np.argmax(self.V)
        ax1.plot(self.states[optimal_idx], self.V[optimal_idx], 'ro', 
                markersize=12, label=f'Optimal State (D={self.states[optimal_idx]:.4f})')
        ax1.set_xlabel('Duty Ratio', fontsize=12)
        ax1.set_ylabel('Value Function', fontsize=12)
        ax1.set_title('Value Function - Dynamic Programming', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # ----- Optimal Policy -----
        # Shows which action to take from each state
        ax2 = plt.subplot(2, 2, 2)
        action_map = [self.delta_actions[a] for a in self.policy]
        ax2.plot(self.states, action_map, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Duty Ratio', fontsize=12)
        ax2.set_ylabel('Optimal Action (ΔD)', fontsize=12)
        ax2.set_title('Optimal Policy - Dynamic Programming', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ----- Convergence Plot -----
        # Shows how quickly value iteration converged
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(range(len(self.value_history)), self.value_history, 'r-', linewidth=2)
        ax3.axhline(y=self.epsilon, color='g', linestyle='--', 
                   label=f'Threshold = {self.epsilon}')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Max Value Change (δ)', fontsize=12)
        ax3.set_title('Convergence - Value Iteration', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')  # Log scale shows convergence better
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # ----- Power Landscape with DP Solution -----
        ax4 = plt.subplot(2, 2, 4)
        D_range = np.linspace(self.D_min, self.D_max, 200)
        P_range = [self.boost.get_input_power(d) for d in D_range]
        
        ax4.plot(D_range, P_range, 'b-', linewidth=2, alpha=0.5, label='Power Landscape')
        
        # Mark discretized states
        state_powers = [self.reward_function(i) for i in range(self.n_states)]
        ax4.scatter(self.states, state_powers, c='cyan', s=20, alpha=0.6, 
                   label='DP States', zorder=3)
        
        # Mark optimal point
        optimal_D = self.states[optimal_idx]
        optimal_P = self.reward_function(optimal_idx)
        ax4.scatter(optimal_D, optimal_P, c='red', s=200, marker='*', 
                   label=f'DP Optimal (D={optimal_D:.4f})', zorder=5)
        
        ax4.set_xlabel('Duty Ratio', fontsize=12)
        ax4.set_ylabel('Power (W)', fontsize=12)
        ax4.set_title('Power Landscape with DP Solution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('dp_results.png', dpi=300, bbox_inches='tight')
        plt.show()


# =================================================================================
# MAIN EXECUTION
# =================================================================================

def main():
    """
    Main execution function
    
    Orchestrates all six optimization tasks:
    1. PV Module Modeling
    2. Perturb & Observe
    3. Non-Linear Optimization
    4. Particle Swarm Optimization
    5. Artificial Neural Network
    6. Dynamic Programming
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "MPPT OPTIMIZATION PROJECT")
    print(" " * 10 + "Boost Converter Duty Ratio Optimization")
    print("=" * 70)
    
    # ========== TASK 1: PV MODULE MODELING ==========
    print("\n" + "=" * 70)
    print("TASK 1: PV MODULE MODELING")
    print("=" * 70)
    
    # Create and characterize PV module
    pv = PVModule()
    V_mpp, I_mpp, P_mpp = pv.plot_characteristics()
    
    # Create boost converter connected to PV module
    boost = BoostConverter(pv)
    
    print(f"\nBoost Converter Parameters:")
    print(f"Output Voltage: {boost.V_out} V")
    print(f"Inductance: {boost.L * 1e3:.2f} mH")
    print(f"Capacitance: {boost.C * 1e6:.2f} µF")
    print(f"Switching Frequency: {boost.f_sw / 1000:.1f} kHz")
    print(f"Efficiency: {boost.efficiency * 100:.1f}%")
    
    # ========== TASK 2: PERTURB AND OBSERVE ==========
    po = PerturbObserve(boost, delta_D=0.005, D_init=0.3)
    po_results = po.run(iterations=100)
    po.plot_results()
    
    # ========== TASK 3: NON-LINEAR OPTIMIZATION ==========
    nlo = NonLinearOptimization(boost)
    nlo_results = nlo.optimize()
    nlo.plot_landscape()
    
    # ========== TASK 4: PARTICLE SWARM OPTIMIZATION ==========
    pso = ParticleSwarmOptimization(boost, n_particles=30, max_iter=50)
    pso_results = pso.optimize()
    pso.plot_results()
    
    # ========== TASK 5: ARTIFICIAL NEURAL NETWORK ==========
    ann = ANNOptimization(boost)
    ann.generate_training_data(n_samples=1000)
    ann.train(epochs=500, learning_rate=0.01, batch_size=32)
    ann.evaluate()
    ann.plot_results()
    
    # ========== TASK 6: DYNAMIC PROGRAMMING ==========
    dp = DynamicProgramming(boost, n_states=100)
    dp_results = dp.value_iteration(max_iterations=1000)
    dp.plot_results()
    
    # =================================================================================
    # SUMMARY COMPARISON OF ALL METHODS
    # =================================================================================
    
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON OF ALL OPTIMIZATION METHODS")
    print("=" * 70)
    
    # Display results in table format
    print("\n{:<30} {:<15} {:<15}".format("Method", "Duty Ratio", "Power (W)"))
    print("-" * 70)
    print("{:<30} {:<15.6f} {:<15.4f}".format("Perturb & Observe", po_results[0], po_results[1]))
    print("{:<30} {:<15.6f} {:<15.4f}".format("Non-Linear (SLSQP)", 
                                               nlo_results[0].x[0], -nlo_results[0].fun))
    print("{:<30} {:<15.6f} {:<15.4f}".format("Non-Linear (L-BFGS-B)", 
                                               nlo_results[1].x[0], -nlo_results[1].fun))
    print("{:<30} {:<15.6f} {:<15.4f}".format("Particle Swarm Opt.", 
                                               pso_results[0], pso_results[1]))
    
    # ANN prediction
    D_ann = ann.predict(pv.T, pv.G, pv.V_mp)
    P_ann = boost.get_input_power(D_ann)
    print("{:<30} {:<15.6f} {:<15.4f}".format("Artificial Neural Net", D_ann, P_ann))
    
    # DP result
    print("{:<30} {:<15.6f} {:<15.4f}".format("Dynamic Programming", 
                                               dp_results[0], dp_results[1]))
    
    print("-" * 70)
    print(f"{'Theoretical Maximum':<30} {'-':<15} {P_mpp:<15.4f}")
    print("=" * 70)
    
    # =================================================================================
    # COMPARATIVE VISUALIZATION
    # =================================================================================
    
    methods = ['P&O', 'SLSQP', 'L-BFGS-B', 'PSO', 'ANN', 'DP']
    powers = [po_results[1], -nlo_results[0].fun, -nlo_results[1].fun, 
              pso_results[1], P_ann, dp_results[1]]
    duty_ratios = [po_results[0], nlo_results[0].x[0], nlo_results[1].x[0], 
                   pso_results[0], D_ann, dp_results[0]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # ----- Power Comparison -----
    ax1.bar(methods, powers, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=P_mpp, color='red', linestyle='--', linewidth=2, 
                label=f'Theoretical Max = {P_mpp:.2f} W')
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Power Comparison - All Methods', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    ax1.set_ylim([0, P_mpp * 1.1])
    
    # ----- Duty Ratio Comparison -----
    ax2.bar(methods, duty_ratios, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Duty Ratio', fontsize=12)
    ax2.set_title('Optimal Duty Ratio - All Methods', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # =================================================================================
    # PROJECT COMPLETION
    # =================================================================================
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("All visualizations have been saved as PNG files.")
    print("=" * 70)


# =================================================================================
# PROGRAM ENTRY POINT
# =================================================================================

if __name__ == "__main__":
    """
    This block ensures main() only runs when script is executed directly,
    not when imported as a module
    """
    main()
