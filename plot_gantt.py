"""
Generates a Gantt chart style comparison of the stage schedules for the
Standard RL and Safe RL agents using the exported evaluation trajectory data (.npz).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def generate_gantt():
    """Generates a Gantt chart comparing stage schedules of Safe vs Standard RL.

    Loads the evaluation trajectory logs for both the Standard RL and the Safe RL
    agents from saved numpy binaries under the policy directory, parses the active
    operational stages over time, and outputs a horizontal bar plot comparing their
    timelines.
    """
    # Load evaluation npz files
    safe_path = os.path.join("policy", "eval_data_safe.npz")
    std_path = os.path.join("policy", "eval_data_standard.npz")
    
    if not os.path.exists(safe_path) or not os.path.exists(std_path):
        print("Error: Make sure both policy/eval_data_safe.npz and policy/eval_data_standard.npz exist.")
        print(f"policy/eval_data_safe.npz exists: {os.path.exists(safe_path)}")
        print(f"policy/eval_data_standard.npz exists: {os.path.exists(std_path)}")
        return

    safe_data = np.load(safe_path)
    std_data = np.load(std_path)

    safe_stages = safe_data["stages"]
    std_stages = std_data["stages"]

    CONTROL_INTERVAL = 10.0
    time_hours = np.arange(len(safe_stages)) * CONTROL_INTERVAL

    # Stage metadata: labels, colors, and descriptions
    stage_colors = {
        0: '#4c72b0',  # Inoculation: muted blue
        1: '#dd8452',  # Growth: muted orange
        2: '#55a868',  # Harvesting: muted green
        3: '#c44e52'   # Idle: muted red
    }
    
    stage_labels = {
        0: 'Inoculation (Stage 0)',
        1: 'Growth (Stage 1)',
        2: 'Harvesting (Stage 2)',
        3: 'Idle (Stage 3)'
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # Helper function to plot stage blocks for an agent
    def plot_agent_timeline(y_pos, stages, agent_label):
        """Plots contiguous stage intervals as horizontal blocks for an agent.

        Groups sequential time steps having the same active stage index into
        single horizontal bars to create a clean Gantt-style timeline.

        Args:
            y_pos (int): The vertical position (y-coordinate) on the chart.
            stages (np.ndarray): Sequence of active stage indices over the episode.
            agent_label (str): Name or description of the agent being plotted.
        """
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
