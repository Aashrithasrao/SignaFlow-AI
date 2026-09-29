"""
Scale-Invariant Geometric Analysis Engine for 50-Class ASL Recognition.

Normalizes all hand and body landmark measurements by:
1. palm_scale: Distance between wrist and middle finger MCP.
2. body_scale: Distance between shoulders (or head-to-torso).

Because all distances and extension ratios are divided by anatomical scales,
the resulting handshapes, spatial zones, and finger extension values are
strictly invariant to camera distance, user hand size, and positioning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def pt_dist(p1, p2) -> float:
    """Euclidean distance in normalized 2D coordinates."""
    return float(((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5)


def analyze_hand_geometry(hand_lms) -> Optional[Dict[str, float]]:
    """
    Extracts scale-invariant geometric metrics from a single hand's landmarks.
    All finger extensions and inter-finger distances are normalized by palm_scale.
    """
    if hand_lms is None or len(hand_lms) < 21:
        return None

    # Palm scale: Wrist (0) to Middle MCP (9)
    palm_scale = pt_dist(hand_lms[0], hand_lms[9])
    if palm_scale < 1e-4:
        return None

    # Finger Extensions (Distance from knuckle MCP to fingertip / palm_scale)
    thumb_ext = pt_dist(hand_lms[2], hand_lms[4]) / palm_scale
    index_ext = pt_dist(hand_lms[5], hand_lms[8]) / palm_scale
    mid_ext = pt_dist(hand_lms[9], hand_lms[12]) / palm_scale
    ring_ext = pt_dist(hand_lms[13], hand_lms[16]) / palm_scale
    pinky_ext = pt_dist(hand_lms[17], hand_lms[20]) / palm_scale

    # Fingertip Distances to Thumb Tip (4) / palm_scale
    d_index_thumb = pt_dist(hand_lms[8], hand_lms[4]) / palm_scale
    d_mid_thumb = pt_dist(hand_lms[12], hand_lms[4]) / palm_scale
    d_ring_thumb = pt_dist(hand_lms[16], hand_lms[4]) / palm_scale
    d_pinky_thumb = pt_dist(hand_lms[20], hand_lms[4]) / palm_scale

    # Spreads between adjacent fingertips
    spread_index_mid = pt_dist(hand_lms[8], hand_lms[12]) / palm_scale
    spread_mid_ring = pt_dist(hand_lms[12], hand_lms[16]) / palm_scale
    spread_ring_pinky = pt_dist(hand_lms[16], hand_lms[20]) / palm_scale

    # Primary Scale-Invariant Hand Shapes
    is_fist = (
        index_ext < 0.72
        and mid_ext < 0.72
        and ring_ext < 0.75
        and pinky_ext < 0.75
    )
    is_duck_beak = (
        index_ext > 0.52
        and mid_ext > 0.52
        and (d_index_thumb < 0.55 or d_mid_thumb < 0.55)
        and pinky_ext < 0.70
    )
    is_pointing = (
        index_ext > 0.78
        and mid_ext < 0.65
        and pinky_ext < 0.68
        and d_index_thumb > 0.65
    )
    is_v_sign = (
        index_ext > 0.72
        and mid_ext > 0.72
        and ring_ext < 0.65
        and pinky_ext < 0.65
        and spread_index_mid > 0.30
    )
    is_w_sign = (
        index_ext > 0.65
        and mid_ext > 0.65
        and ring_ext > 0.65
        and pinky_ext < 0.68
        and d_pinky_thumb < 0.60
    )
    is_flat_open = (
        index_ext > 0.68
        and mid_ext > 0.68
        and ring_ext > 0.68
        and pinky_ext > 0.68
        and d_index_thumb > 0.40
    )
    is_y_hand = (
        thumb_ext > 0.75
        and pinky_ext > 0.75
        and mid_ext < 0.65
        and ring_ext < 0.65
    )
    is_l_hand = (
        thumb_ext > 0.75
        and index_ext > 0.75
        and mid_ext < 0.65
        and ring_ext < 0.65
    )
    is_pinched = (
        d_index_thumb < 0.55
        and d_mid_thumb < 0.55
        and d_ring_thumb < 0.65
    )
    is_c_hand = (
        0.50 <= index_ext <= 0.85
        and 0.50 <= mid_ext <= 0.85
        and 0.45 <= d_index_thumb <= 0.90
    )

    return {
        "palm_scale": palm_scale,
        "wrist_x": hand_lms[0].x,
        "wrist_y": hand_lms[0].y,
        "mid_mcp_x": hand_lms[9].x,
        "mid_mcp_y": hand_lms[9].y,
        "index_ext": index_ext,
        "mid_ext": mid_ext,
        "ring_ext": ring_ext,
        "pinky_ext": pinky_ext,
        "thumb_ext": thumb_ext,
        "d_index_thumb": d_index_thumb,
        "d_mid_thumb": d_mid_thumb,
        "d_ring_thumb": d_ring_thumb,
        "d_pinky_thumb": d_pinky_thumb,
        "spread_index_mid": spread_index_mid,
        "spread_mid_ring": spread_mid_ring,
        "spread_ring_pinky": spread_ring_pinky,
        "is_fist": is_fist,
        "is_duck_beak": is_duck_beak,
        "is_pointing": is_pointing,
        "is_v_sign": is_v_sign,
        "is_w_sign": is_w_sign,
        "is_flat_open": is_flat_open,
        "is_y_hand": is_y_hand,
        "is_l_hand": is_l_hand,
        "is_pinched": is_pinched,
        "is_c_hand": is_c_hand,
    }


def analyze_spatial_zones(hand_geo: Dict, pose_lms) -> Dict[str, bool]:
    """
    Computes scale-invariant spatial zones (forehead, mouth/chin, ear, chest, neutral).
    All boundary thresholds are normalized by shoulder_scale.
    """
    if pose_lms is None or len(pose_lms) < 13:
        # Default neutral if pose unavailable
        return {
            "is_forehead": False,
            "is_mouth_chin": False,
            "is_ear_temple": False,
            "is_chest": True,
            "is_neutral": True,
        }

    nose = pose_lms[0]
    left_sh = pose_lms[11]
    right_sh = pose_lms[12]

    shoulder_scale = pt_dist(left_sh, right_sh)
    if shoulder_scale < 1e-3:
        shoulder_scale = hand_geo["palm_scale"] * 2.5

    mid_sh_x = (left_sh.x + right_sh.x) / 2.0
    mid_sh_y = (left_sh.y + right_sh.y) / 2.0

    chin_y = nose.y + 0.35 * (mid_sh_y - nose.y)
    forehead_y = nose.y - 0.20 * (mid_sh_y - nose.y)
    chest_bottom_y = mid_sh_y + 0.85 * shoulder_scale

    hand_x = hand_geo["wrist_x"]
    hand_y = hand_geo["wrist_y"]

    is_forehead = (hand_y <= chin_y) and (hand_y <= forehead_y + 0.15 * shoulder_scale)
    is_mouth_chin = (
        (hand_y > forehead_y)
        and (hand_y <= chin_y + 0.18 * shoulder_scale)
        and (abs(hand_x - nose.x) < 0.45 * shoulder_scale)
    )
    is_ear_temple = (
        (hand_y <= chin_y + 0.15 * shoulder_scale)
        and (abs(hand_x - nose.x) >= 0.30 * shoulder_scale)
    )
    is_chest = (
        (chin_y < hand_y <= chest_bottom_y)
        and (abs(hand_x - mid_sh_x) < 0.70 * shoulder_scale)
    )
    is_neutral = hand_y > chin_y

    return {
        "is_forehead": is_forehead,
        "is_mouth_chin": is_mouth_chin,
        "is_ear_temple": is_ear_temple,
        "is_chest": is_chest,
        "is_neutral": is_neutral,
    }


def apply_scale_invariant_corroboration(
    probabilities: np.ndarray,
    classes: List[str],
    result
) -> np.ndarray:
    """
    Applies scale-invariant geometric corroboration across all 50 ASL vocabulary classes.
    Adjusts raw model probabilities with anatomical boundary constraints.
    """
    if result is None:
        return probabilities

    # Collect active hands
    hands = [h for h in [result.right_hand_landmarks, result.left_hand_landmarks] if h]
    if not hands:
        return probabilities

    probs = probabilities.copy()
    num_hands = len(hands)

    # Analyze first dominant hand
    hand_geo = analyze_hand_geometry(hands[0])
    if hand_geo is None:
        return probabilities

    zones = analyze_spatial_zones(hand_geo, result.pose_landmarks)

    def get_idx(sign_name: str) -> int:
        return classes.index(sign_name) if sign_name in classes else -1

    # Indices for critical signs
    yes_idx = get_idx("yes")
    no_idx = get_idx("no")
    go_idx = get_idx("go")
    please_idx = get_idx("please")
    dad_idx = get_idx("dad")
    mom_idx = get_idx("mom")
    water_idx = get_idx("water")
    food_idx = get_idx("food")
    drink_idx = get_idx("drink")
    thankyou_idx = get_idx("thankyou")
    hello_idx = get_idx("hello")
    hear_idx = get_idx("hear")
    listen_idx = get_idx("listen")
    hungry_idx = get_idx("hungry")
    thirsty_idx = get_idx("thirsty")
    fine_idx = get_idx("fine")
    happy_idx = get_idx("happy")
    sad_idx = get_idx("sad")
    boy_idx = get_idx("boy")
    girl_idx = get_idx("girl")
    now_idx = get_idx("now")
    stay_idx = get_idx("stay")
    same_idx = get_idx("same")
    later_idx = get_idx("later")
    look_idx = get_idx("look")
    see_idx = get_idx("see")
    have_idx = get_idx("have")
    book_idx = get_idx("book")
    close_idx = get_idx("close")
    open_idx = get_idx("open")
    wait_idx = get_idx("wait")

    # =========================================================================
    # 1. NEUTRAL SPACE GESTURES (YES, NO, GO, NOW, STAY, SAME, LATER, LOOK/SEE)
    # =========================================================================

    # YES (Closed S-fist in neutral space)
    if hand_geo["is_fist"]:
        if yes_idx >= 0:
            probs[yes_idx] = max(probs[yes_idx] * 2.5, 0.50)
            if no_idx >= 0: probs[no_idx] *= 0.08
            if go_idx >= 0: probs[go_idx] *= 0.08
            if please_idx >= 0: probs[please_idx] *= 0.08

    # NO (Duck-beak pinch snapping onto thumb)
    elif hand_geo["is_duck_beak"]:
        if no_idx >= 0:
            probs[no_idx] = max(probs[no_idx] * 2.5, 0.50)
            if yes_idx >= 0: probs[yes_idx] *= 0.08
            if go_idx >= 0: probs[go_idx] *= 0.08
            if please_idx >= 0: probs[please_idx] *= 0.08

    # GO (Index extended, middle/pinky curled, index far from thumb)
    elif hand_geo["is_pointing"] and not zones["is_ear_temple"] and not zones["is_mouth_chin"]:
        if go_idx >= 0:
            probs[go_idx] = max(probs[go_idx] * 2.5, 0.50)
            if no_idx >= 0: probs[no_idx] *= 0.08
            if yes_idx >= 0: probs[yes_idx] *= 0.08
            if please_idx >= 0: probs[please_idx] *= 0.08

    # LOOK / SEE (V-peace sign pointing away from eyes)
    elif hand_geo["is_v_sign"]:
        if see_idx >= 0: probs[see_idx] = max(probs[see_idx] * 2.0, 0.45)
        if look_idx >= 0: probs[look_idx] = max(probs[look_idx] * 2.0, 0.45)
        if no_idx >= 0: probs[no_idx] *= 0.10
        if yes_idx >= 0: probs[yes_idx] *= 0.10

    # NOW / STAY / SAME (Y-handshape: thumb & pinky extended)
    elif hand_geo["is_y_hand"]:
        for y_idx in [now_idx, stay_idx, same_idx]:
            if y_idx >= 0:
                probs[y_idx] = max(probs[y_idx] * 2.0, 0.40)
        if yes_idx >= 0: probs[yes_idx] *= 0.10
        if no_idx >= 0: probs[no_idx] *= 0.10

    # LATER (L-handshape: thumb & index at 90 deg)
    elif hand_geo["is_l_hand"]:
        if later_idx >= 0:
            probs[later_idx] = max(probs[later_idx] * 2.2, 0.45)

    # =========================================================================
    # 2. FOREHEAD ZONE GESTURES (DAD, BOY, HELLO)
    # =========================================================================
    if zones["is_forehead"]:
        # DAD: Open 5-hand at forehead
        if hand_geo["is_flat_open"] and dad_idx >= 0:
            probs[dad_idx] = max(probs[dad_idx] * 2.5, 0.50)
            if mom_idx >= 0: probs[mom_idx] *= 0.05
            if please_idx >= 0: probs[please_idx] *= 0.05
        # BOY: Pinched near forehead
        elif (hand_geo["is_pinched"] or hand_geo["is_duck_beak"]) and boy_idx >= 0:
            probs[boy_idx] = max(probs[boy_idx] * 2.2, 0.45)
        # HELLO: Open hand near temple/forehead
        elif hand_geo["is_flat_open"] and hello_idx >= 0:
            probs[hello_idx] = max(probs[hello_idx] * 2.0, 0.45)

    # =========================================================================
    # 3. MOUTH & CHIN ZONE GESTURES (MOM, WATER, FOOD, DRINK, THANKYOU, THIRSTY)
    # =========================================================================
    elif zones["is_mouth_chin"]:
        # MOM: Open 5-hand touching chin/jaw
        if hand_geo["is_flat_open"] and mom_idx >= 0:
            probs[mom_idx] = max(probs[mom_idx] * 2.5, 0.50)
            if dad_idx >= 0: probs[dad_idx] *= 0.05
            if please_idx >= 0: probs[please_idx] *= 0.05
        # WATER: "W" 3-finger shape tapping chin
        elif hand_geo["is_w_sign"] and water_idx >= 0:
            probs[water_idx] = max(probs[water_idx] * 2.5, 0.55)
            if food_idx >= 0: probs[food_idx] *= 0.10
            if drink_idx >= 0: probs[drink_idx] *= 0.10
        # FOOD: Pinched fingertips to mouth
        elif hand_geo["is_pinched"] and food_idx >= 0:
            probs[food_idx] = max(probs[food_idx] * 2.5, 0.55)
            if water_idx >= 0: probs[water_idx] *= 0.10
        # DRINK: Cup / C-hand near mouth
        elif hand_geo["is_c_hand"] and drink_idx >= 0:
            probs[drink_idx] = max(probs[drink_idx] * 2.2, 0.45)
        # THANKYOU: Flat hand from chin forward
        elif hand_geo["is_flat_open"] and thankyou_idx >= 0:
            probs[thankyou_idx] = max(probs[thankyou_idx] * 2.0, 0.45)
        # THIRSTY: Pointing index tracing down throat
        elif hand_geo["is_pointing"] and thirsty_idx >= 0:
            probs[thirsty_idx] = max(probs[thirsty_idx] * 2.4, 0.50)

    # =========================================================================
    # 4. EAR / TEMPLE ZONE (HEAR, LISTEN)
    # =========================================================================
    elif zones["is_ear_temple"]:
        if hand_geo["is_pointing"] and hear_idx >= 0:
            probs[hear_idx] = max(probs[hear_idx] * 2.5, 0.50)
            if go_idx >= 0: probs[go_idx] *= 0.10
        elif listen_idx >= 0:
            probs[listen_idx] = max(probs[listen_idx] * 2.2, 0.45)

    # =========================================================================
    # 5. CHEST ZONE (PLEASE, HAPPY, HUNGRY, FINE, HAVE)
    # =========================================================================
    elif zones["is_chest"]:
        # PLEASE: Open palm on chest
        if hand_geo["is_flat_open"]:
            if please_idx >= 0:
                probs[please_idx] = max(probs[please_idx] * 2.5, 0.50)
                if dad_idx >= 0: probs[dad_idx] *= 0.05
                if mom_idx >= 0: probs[mom_idx] *= 0.05
                if no_idx >= 0: probs[no_idx] *= 0.08
                if yes_idx >= 0: probs[yes_idx] *= 0.08
            if happy_idx >= 0:
                probs[happy_idx] = max(probs[happy_idx] * 1.8, 0.40)
            if fine_idx >= 0:
                probs[fine_idx] = max(probs[fine_idx] * 1.8, 0.40)
        # HUNGRY: C-hand sliding down chest
        elif hand_geo["is_c_hand"] and hungry_idx >= 0:
            probs[hungry_idx] = max(probs[hungry_idx] * 2.2, 0.45)

    # =========================================================================
    # 6. TWO-HANDED GESTURES (BOOK, CLOSE, OPEN, WAIT)
    # =========================================================================
    if num_hands >= 2:
        for idx in [book_idx, close_idx, open_idx, wait_idx]:
            if idx >= 0:
                probs[idx] = max(probs[idx] * 1.8, 0.38)

    # Normalize softmax probabilities to sum to 1.0
    sum_p = np.sum(probs)
    if sum_p > 1e-6:
        probs = probs / sum_p

    return probs
