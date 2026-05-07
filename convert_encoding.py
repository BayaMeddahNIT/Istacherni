import sys

def convert_to_utf8(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-16') as f:
            content = f.read()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    convert_to_utf8('unknown_articles_report.txt', 'unknown_articles_report_utf8.txt')
