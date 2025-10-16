#!/usr/bin/env python3
"""
Analyze mm_actions.json statistics:
- Count total actions
- Find duplicate/unique actions
- Categorize actions into groups
"""
import json
from collections import Counter
from typing import Dict, List, Set

# Action categorization
ACTION_GROUPS = {
    "Arm Motion": [
        "left upper limb extension", "right upper limb extension", "both upper limb extension",
        "Fist Punch", "Palm Stop", "One arm front raise and the other arm side raise",
        "Arms Circle Motion", "Arms Freestyle Swim", "Hands to Head", "Both Hands Forward",
        "One Arm Forward", "Hand Rotation", "both arms Lift to Chest", "both arms Lift Overhead",
        "one arm to Head", "one arm Lift Overhead", "Chest Expansion (Vertical)", "Chest Expansion (Horizontal)",
        "Chest expansion(horizontal)", "Chest expansion(vertical)", "One Arm T-Pose", "Arms Open Stand", "One Hand to Head",
        # All hand/arm related actions
        "Arm Curl", "Arm Forward", "Arm Swing", "Arms Flap", "Arms Freestyle Motion", "Arms Open Turn",
        "Hand Raise", "Hands Knead", "Hands Mop Clean", "Hands on Hips", "Hands on Knee",
        "Natural Arm Swing", "One Arm Handshake", "One Arm Pull", "One Arm Swing",
        "One Hand Pour", "One Hand Raise", "One Hand Salt Toss", "One Hand Slap",
        "One Hand Stir", "One Hand Sweep", "One Hand Swirl", "One Hand to head",
        "Overhead Press", "Overhead Throw", "Pan Cooking Motion",
        "Right Hand Beckon Wave", "Right Hand Dig (Fan Turn)", "Scoop Tap Motion",
        "Side Lateral Raise", "Swimming Hands (Movement)",
        "Both Hands Pour", "Both Hands Stir", "Both upper-limb lateral extension",
        "Single hand forward raise", "Single upper-limb lateral extension", "Single-hand lateral raise",
        "Unilateral upper-lower limb extension",
        # From Sports (arm-focused)
        "Basketball Dribble", "Basketball Shot", "Hands Toss Pose (Volleyball)",
        "Boxing Guard", "Badminton Swing", "Badminton Under Serve",
        "Golf Swing", "Underhand Volleyball Serve", "Volleyball Spike",
        "Throwing",
        # From Daily-life (arm/hand gestures)
        "Hands Prayer Pose", "Pour Water Motion", "Cup Drinking Pose",
        "Arm Handshake", "Picking up things", "Bowl Eating Motion", "x",
    ],
    "Leg Motion": [
        "left front lunge", "right front lunge", "left side lunge", "right side lunge",
        "left limb extension", "right limb",
        "Single Leg Jump", "Wide Leg Jump", "Front Kick", "Side Kick", "Inside Kick",
        "Front Lunge", "Knee Fold", "Knee Up", "One Leg Stand",
        # All lunge actions
        "Front lunge", "Side lunge",
        "Kicking", "Leg Stand", "High Knee Run", "Straight Knee Hop",
        # From Sports (leg-focused)
        "Goalkeeper Defense Stance",
    ],
    "Torso Movement": [
        "squat", "Squat",
        "Sway", "Twist", "Bow", "Lean Back", "Diagonal stand", "Diagonal Stand",
        "Torso Tilt", "Head move",
        "Bowing", "Feet Moderate Wide Stance", "Feet Shoulder Width", "Feet Wide Stance",
        "Pivot Turn", "Hands on Hips", "Hands on Knee", "Wide Stance Palms forward",
        # From Sports (torso-focused)
        "Bench Press", "Barbell Row", "Pull Up",
        # From Daily-life
        "Hug",
    ],
    "Full Body Motion": [
        "Exaggerated March Step", "Jogging", "In-place high knee run",
        "Jumping Up", "Jumping UP", "Movement Jumping up",
        "Walk Forward", "Walk Backward", "Side Step",
        "T-Pose", "Attention",
        "Jumping up", "Walk", "Boxing Step and Punch",
        # From Daily-life
        "Mark Time", "Mark time",
    ],
}


def load_actions(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def categorize_action(action: str) -> str:
    """Categorize an action into a group."""
    for group, actions in ACTION_GROUPS.items():
        if action in actions:
            return group
    return "Uncategorized"


def analyze_stats(json_path: str) -> None:
    data = load_actions(json_path)
    
    print("="*80)
    print("mm_actions.json Statistics")
    print("="*80)
    
    # Basic counts
    total_records = len(data)
    print(f"\nTotal records: {total_records}")
    
    # Source distribution
    sources = Counter(rec.get("info", {}).get("from", "unknown") for rec in data)
    print(f"\nRecords by source:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt}")
    
    # Action statistics
    actions = [rec.get("action", "") for rec in data if rec.get("action")]
    action_counts = Counter(actions)
    unique_actions = len(action_counts)
    
    print(f"\nTotal action instances: {len(actions)}")
    print(f"Unique actions: {unique_actions}")
    
    # Most common actions
    print(f"\nTop 20 most frequent actions:")
    for action, count in action_counts.most_common(20):
        print(f"  {action}: {count}")
    
    # Duplicate analysis (actions appearing more than once)
    duplicates = {action: count for action, count in action_counts.items() if count > 1}
    print(f"\nActions appearing multiple times: {len(duplicates)}")
    print(f"Actions appearing only once: {unique_actions - len(duplicates)}")
    
    # Categorization
    categorized: Dict[str, List[str]] = {group: [] for group in ACTION_GROUPS.keys()}
    categorized["Uncategorized"] = []
    
    action_to_group: Dict[str, str] = {}
    for action in action_counts.keys():
        group = categorize_action(action)
        categorized[group].append(action)
        action_to_group[action] = group
    
    print("\n" + "="*80)
    print("Action Categorization")
    print("="*80)
    
    for group in list(ACTION_GROUPS.keys()) + ["Uncategorized"]:
        actions_in_group = categorized[group]
        if not actions_in_group:
            continue
        
        # Count instances (not unique actions)
        instance_count = sum(action_counts[act] for act in actions_in_group)
        
        print(f"\n{group}: {len(actions_in_group)} unique actions, {instance_count} instances")
        for action in sorted(actions_in_group):
            count = action_counts[action]
            print(f"  - {action}: {count}")
    
    # Movement statistics
    movement_true = sum(1 for rec in data if rec.get("movement", False))
    movement_false = total_records - movement_true
    print(f"\n" + "="*80)
    print("Movement Statistics")
    print("="*80)
    print(f"Records with movement=true: {movement_true} ({movement_true/total_records*100:.1f}%)")
    print(f"Records with movement=false: {movement_false} ({movement_false/total_records*100:.1f}%)")
    
    # Segment statistics
    segments = Counter(rec.get("info", {}).get("segment", "unknown") for rec in data)
    print(f"\n" + "="*80)
    print("Segment Statistics")
    print("="*80)
    print(f"Total segments: {len(segments)}")
    print(f"Actions per segment (avg): {total_records / len(segments):.1f}")
    
    # Export categorization mapping
    output_path = json_path.replace(".json", "_categories.json")
    category_export = {
        "action_to_group": action_to_group,
        "group_summary": {
            group: {
                "unique_actions": len(actions_in_group),
                "total_instances": sum(action_counts[act] for act in actions_in_group),
                "actions": sorted(actions_in_group)
            }
            for group, actions_in_group in categorized.items()
            if actions_in_group
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(category_export, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Category mapping exported to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/mmWave-QA/mmWaveQA_benchmark/mm_actions.json"
    analyze_stats(json_path)

