"""
Generates a Gantt chart style comparison of the stage schedules for the
Standard RL and Safe RL agents using the exported evaluation trajectory data (.npz).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def generate_gantt():
    # Load evaluation npz files
    safe_path = "eval_data_safe.npz"
    std_path = "eval_data_standard.npz"
    
    if not os.path.exists(safe_path) or not os.path.exists(std_path):
        print("Error: Make sure both eval_data_safe.npz and eval_data_standard.npz exist.")
        print(f"eval_data_safe.npz exists: {os.path.exists(safe_path)}")
        print(f"eval_data_standard.npz exists: {os.path.exists(std_path)}")
        return

    safe_data = np.load(safe_path)
    std_data = np.load(std_path)

    safe_stages = safe_data["stages"]
    std_stages = std_data["stages"]

    CONTROL_INTERVAL = 10.0
    time_hours = np.arange(len(safe_stages)) * CONTROL_INTERVAL

    # Stage metadata: labels, colors, and descriptions
    stage_colors = {
        0: '#4c72b0',  # Growth: muted blue
        1: '#dd8452',  # Production: muted orange
        2: '#55a868',  # Cleanup: muted green
        3: '#c44e52'   # Idle: muted red
    }
    
    stage_labels = {
        0: 'Growth (Stage 0)',
        1: 'Production (Stage 1)',
        2: 'Cleanup (Stage 2)',
        3: 'Idle (Stage 3)'
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # Helper function to plot stage blocks for an agent
    def plot_agent_timeline(y_pos, stages, agent_label):
        # We find contiguous stage segments to plot them as single bars
        start_idx = 0
        current_stage = stages[0]
        
        for i in range(1, len(stages)):
            if stages[i] != current_stage or i == len(stages) - 1:
                end_idx = i if stages[i] != current_stage else i + 1
                start_time = start_idx * CONTROL_INTERVAL
                end_time = end_idx * CONTROL_INTERVAL
                duration = end_time - start_time
                
                # Draw the bar segment
                ax.barh(y_pos, duration, left=start_time, height=0.4, 
                        color=stage_colors[current_stage], edgecolor='black', alpha=0.95)
                
                # Add stage numbers inside the blocks if they are wide enough
                if duration > 30:
                    ax.text(start_time + duration/2, y_pos, f"S{current_stage}", 
                            ha='center', va='center', color='white', fontweight='bold', fontsize=10)
                
                start_idx = i
                current_stage = stages[i]

    # Plot both timelines (Safe RL at y=1, Standard RL at y=2)
    plot_agent_timeline(1, safe_stages, "Safe RL")
    plot_agent_timeline(2, std_stages, "Standard RL")

    # Set up layout, ticks, and labels
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Safe RL", "Standard RL"], fontweight='bold')
    ax.set_xlabel("Time (hours)", fontsize=12, fontweight='bold')
    ax.set_title("Operational Stage Schedule Comparison (Best Episode Run)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1000)
    
    # Grid and ticks
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.grid(False)

    # Create manual legend patches
    legend_patches = [
        mpatches.Patch(color=stage_colors[s], label=stage_labels[s], edgecolor='black')
        for s in range(4)
    ]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=True, facecolor='white', edgecolor='gray')

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    os.makedirs("plot", exist_ok=True)
    out_path = os.path.join("plot", "gantt_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Gantt comparison plot successfully saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    generate_gantt()
