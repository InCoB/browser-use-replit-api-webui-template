# fix_indentation.py
import os

target_file = "api/app.py"
lines_to_fix = {
    '    app.logger.debug(f"File descriptor limit set to {hard}")': 'app.logger.debug(f"File descriptor limit set to {hard}")',
    '    app.logger.debug(f"Process limit set to {hard}")': 'app.logger.debug(f"Process limit set to {hard}")',
    '        app.logger.warning(f"Could not increase resource limits: {str(e)}")': '    app.logger.warning(f"Could not increase resource limits: {str(e)}")',
}
fixed_lines_count = 0

print(f"Attempting to fix indentation in {target_file}...")

try:
    with open(target_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        # Strip trailing newline for comparison, but keep original ending
        line_content = line.rstrip("\n")
        original_ending = line[len(line_content) :]  # Capture original newline chars

        found_fix = False
        for incorrect_line, correct_line_content in lines_to_fix.items():
            # Check if the line content matches one of the known incorrect lines
            if line_content.strip() == incorrect_line.strip():
                # Check if the indentation is actually wrong
                if line_content.startswith(
                    incorrect_line.split(incorrect_line.strip())[0]
                ):
                    # Find the correct indentation from the correct version
                    correct_indent = correct_line_content.split(
                        correct_line_content.strip()
                    )[0]
                    corrected_line_with_indent = correct_indent + line_content.strip()

                    print(f"  - Fixing line {i+1}: Found incorrect indentation.")
                    print(f"    Original: '{line_content}'")
                    print(f"    Corrected: '{corrected_line_with_indent}'")
                    new_lines.append(corrected_line_with_indent + original_ending)
                    fixed_lines_count += 1
                    found_fix = True
                    break  # Move to next line once fixed

        if not found_fix:
            new_lines.append(line)  # Keep the original line if no fix needed

    if fixed_lines_count > 0:
        print(
            f"\nWriting {fixed_lines_count} corrected line(s) back to {target_file}..."
        )
        with open(target_file, "w") as f:
            f.writelines(new_lines)
        print("Indentation fix applied successfully.")
    else:
        print("No incorrect indentation found matching the specific lines.")

except FileNotFoundError:
    print(f"Error: File not found at {target_file}")
except Exception as e:
    print(f"An error occurred: {e}")
