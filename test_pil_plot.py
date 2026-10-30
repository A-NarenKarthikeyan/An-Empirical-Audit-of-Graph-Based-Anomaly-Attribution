from PIL import Image, ImageDraw, ImageFont
import os

def render_abstention_tradeoff_plot(
    abstain_rates, c_faith_rates, v_faith_rates, tau_values,
    current_tau=0.70, output_path="results/abstention_tradeoff.png"
):
    width, height = 900, 600
    margin_left, margin_right = 100, 50
    margin_top, margin_bottom = 80, 80

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Gridlines & Y-axis ticks (0% to 100% Faithfulness)
    for y_val in range(0, 101, 10):
        y_pos = margin_top + plot_h - int((y_val / 100.0) * plot_h)
        draw.line([(margin_left, y_pos), (margin_left + plot_w, y_pos)], fill=(230, 230, 230), width=1)
        draw.text((margin_left - 45, y_pos - 6), f"{y_val}%", fill=(80, 80, 80))

    # Gridlines & X-axis ticks (0% to 100% Abstention Rate)
    for x_val in range(0, 101, 10):
        x_pos = margin_left + int((x_val / 100.0) * plot_w)
        draw.line([(x_pos, margin_top), (x_pos, margin_top + plot_h)], fill=(230, 230, 230), width=1)
        draw.text((x_pos - 12, margin_top + plot_h + 10), f"{x_val}%", fill=(80, 80, 80))

    # Axes
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)], fill=(0, 0, 0), width=2)
    draw.line([(margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)], fill=(0, 0, 0), width=2)

    # Title & Axis Labels
    draw.text((margin_left + 120, margin_top - 50), "Abstention vs. Faithfulness Trade-Off Curve", fill=(0, 0, 0))
    draw.text((margin_left + plot_w // 2 - 80, margin_top + plot_h + 40), "Abstention Rate (%)", fill=(0, 0, 0))
    draw.text((margin_left - 90, margin_top + plot_h // 2), "Faithfulness\nRate (%)", fill=(0, 0, 0))

    # Map Points
    pts_c = []
    pts_v = []
    current_x = None
    current_y_c = None
    current_y_v = None

    for x_pct, c_val, v_val, tau in zip(abstain_rates, c_faith_rates, v_faith_rates, tau_values):
        px = margin_left + int(x_pct * plot_w)
        py_c = margin_top + plot_h - int(c_val * plot_h)
        py_v = margin_top + plot_h - int(v_val * plot_h)

        pts_c.append((px, py_c))
        pts_v.append((px, py_v))

        if abs(tau - current_tau) < 1e-4:
            current_x = px
            current_y_c = py_c
            current_y_v = py_v

    # Draw Lines
    for i in range(len(pts_c) - 1):
        draw.line([pts_c[i], pts_c[i+1]], fill=(31, 119, 180), width=3) # Blue for Structural
        draw.line([pts_v[i], pts_v[i+1]], fill=(44, 160, 44), width=3)  # Green for Evidential

    # Draw Points
    for px, py in pts_c:
        draw.ellipse([px-4, py-4, px+4, py+4], fill=(31, 119, 180), outline=(0, 0, 0))
    for px, py in pts_v:
        draw.ellipse([px-4, py-4, px+4, py+4], fill=(44, 160, 44), outline=(0, 0, 0))

    # Mark Current Threshold (tau = 0.70)
    if current_x is not None:
        draw.line([(current_x, margin_top), (current_x, margin_top + plot_h)], fill=(214, 39, 40), width=2)
        draw.text((current_x + 8, margin_top + 20), f"Current Config Threshold (min_topk_mass = {current_tau:.2f})", fill=(214, 39, 40))

    # Legend
    leg_x = margin_left + plot_w - 320
    leg_y = margin_top + 20
    draw.rectangle([leg_x, leg_y, leg_x + 300, leg_y + 70], fill=(250, 250, 250), outline=(200, 200, 200))
    draw.line([(leg_x + 15, leg_y + 20), (leg_x + 45, leg_y + 20)], fill=(31, 119, 180), width=3)
    draw.text((leg_x + 55, leg_y + 12), "Structural Faithfulness (Swap)", fill=(0, 0, 0))

    draw.line([(leg_x + 15, leg_y + 50), (leg_x + 45, leg_y + 50)], fill=(44, 160, 44), width=3)
    draw.text((leg_x + 55, leg_y + 42), "Evidential Sensitivity (Value)", fill=(0, 0, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Plot saved to {output_path}")

# Test plot generation
x_test = [0.107, 0.107, 0.107, 0.107, 0.357, 0.607, 0.750, 0.786]
tau_test = [0.00, 0.20, 0.40, 0.70, 0.75, 0.80, 0.85, 0.95]
c_test = [0.80, 0.80, 0.80, 0.80, 0.778, 0.636, 0.714, 0.667]
v_test = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

render_abstention_tradeoff_plot(x_test, c_test, v_test, tau_test)
