import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches

def create_figure():
    # Steps data
    steps = [
        {"title": "Phase 1: Data Collection", "color": "#e3f2fd"},
        {"title": "Phase 2: Preprocessing", "color": "#fff3e0"},
        {"title": "Phase 3: Imbalance Handling", "color": "#e8f5e9"},
        {"title": "Phase 4: Data Splitting", "color": "#f3e5f5"},
        {"title": "Phase 5: Model Training", "color": "#e0f7fa"},
        {"title": "Phase 6: Evaluation", "color": "#fff9c4"},
        {"title": "Phase 7: Analysis & Deployment", "color": "#fbe9e7"}
    ]

    # Figure setup - Reduced size
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Constants
    box_x = 20
    box_width = 70
    box_height = 8
    gap = 4
    start_y = 85
    circle_x = 10

    # Draw each step
    for i, step in enumerate(steps):
        y_pos = start_y - i * (box_height + gap)
        
        # 1. Number Circle
        circle = patches.Circle((circle_x, y_pos + box_height/2), radius=2.5, 
                                facecolor='#1976d2', edgecolor='black', linewidth=1, zorder=3)
        ax.add_patch(circle)
        ax.text(circle_x, y_pos + box_height/2, str(i+1), 
                ha='center', va='center', color='white', fontsize=12, fontweight='bold', zorder=4)
        
        # 2. Main Box (Rounded Rectangle)
        # Using FancyBboxPatch for rounded corners
        rect = patches.FancyBboxPatch((box_x, y_pos), box_width, box_height,
                                      boxstyle="round,pad=0.2", 
                                      facecolor=step['color'], edgecolor='black', linewidth=1, zorder=2)
        ax.add_patch(rect)
        
        # 3. Text (Title only, centered)
        # Calculate center of box for text placement
        # Text is placed at box_x + box_width/2, y_pos + box_height/2
        ax.text(box_x + box_width/2, y_pos + box_height/2, step['title'], 
                ha='center', va='center', fontsize=12, fontweight='bold', color='#333333', zorder=5)

        # 4. Connecting Arrow (except last one)
        if i < len(steps) - 1:
            arrow_x = circle_x
            # Arrow starts from bottom of current circle
            arrow_y_start = y_pos + box_height/2 - 2.5
            # Arrow ends at top of next circle
            arrow_y_end = (y_pos - gap) + box_height/2 + 2.5
            
            # Simple arrow just connecting the circles
            ax.annotate("", xy=(arrow_x, arrow_y_end), xytext=(arrow_x, arrow_y_start),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5), zorder=1)

    # Main Title
    ax.text(50, 96, "Proposed Aspect-Based Sentiment Analysis Pipeline", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('methodology_flow.pdf', bbox_inches='tight')
    print("Image saved as methodology_flow.pdf")

if __name__ == "__main__":
    create_figure()
