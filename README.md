# 🔆 Hybrid PSO–INC Based MPPT for PV System

## 📌 Overview

This project implements a **Hybrid Maximum Power Point Tracking (MPPT)** algorithm for a photovoltaic (PV) system interfaced with a **DC–DC boost converter**.

The proposed method combines:

- **Particle Swarm Optimization (PSO)** for global maximum power point (GMPP) detection
- **Incremental Conductance (INC)** for local fine tracking

The hybrid approach improves tracking accuracy and reduces steady-state oscillations, especially under **partial shading conditions**, where the PV power–voltage (P–V) curve exhibits multiple local maxima.

---

## ⚡ System Description

The system consists of:

PV Array → Boost Converter → Load

The boost converter duty cycle controls the PV operating voltage:

Vpv = Vo (1 − D)

By adjusting the duty cycle (D), the PV operating point is shifted along the P–V curve to extract maximum power.

---

## 🎯 Problem Statement

Under uniform irradiance, a PV system has a single maximum power point (MPP).

Under partial shading:
- The P–V curve becomes multi-modal (multiple peaks)
- Classical MPPT methods (e.g., INC) may converge to a local maximum

This project addresses the limitation by:
- Formulating MPPT as a nonlinear optimization problem
- Using PSO for global search
- Using INC for local refinement

---

## 🧠 Algorithm Workflow

### 1️⃣ Pure PSO
- Duty cycle treated as optimization variable
- Global search for maximum power
- Effective under multi-peak conditions

### 2️⃣ Pure INC
- Uses slope condition (dP/dV = 0 at MPP)
- Fast local convergence
- May fail under partial shading

### 3️⃣ Hybrid PSO + INC (Proposed Method)
- PSO locates global MPP region
- INC refines the operating point locally
- Reduces oscillations and improves precision

---

## 📊 Features

- Multi-peak P–V curve modeling under partial shading
- Global optimization using PSO
- Local refinement using Incremental Conductance
- Comparative analysis: PSO vs INC vs Hybrid
- Convergence visualization
- Power–Voltage characteristic plotting

---

## 🛠 Implementation Details

- Language: Python
- Libraries: NumPy, Matplotlib
- PV Model: Single-diode approximation
- Converter Model: Ideal boost converter relation

---

## 📈 Results

The hybrid method demonstrates:

- Improved global MPP detection
- Faster convergence compared to pure PSO
- Reduced steady-state oscillations compared to pure INC
- Robust performance under partial shading

---

## 🚀 How to Run

1. Clone the repository:
git clone <https://github.com/rajeev-333/-Hybrid-PSO-INC-Based-MPPT-for-PV-System>
cd <Hybrid PSO–INC Based MPPT for PV System>


2. Install dependencies:


pip install numpy matplotlib


3. Run the script:


python mppt_hybrid.py


The program will generate:
- P–V curve with tracked operating points
- Convergence plots
- Power comparison results

---

