"""Script that dumps the question contents of a JSON file to stdout"""
import json
import argparse

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Dump the question contents of a JSON file to stdout")
    parser.add_argument("path_to_json_file", help="The JSON file to dump")
    args = parser.parse_args()
    with open(args.path_to_json_file) as json_file:
        json_data = json.load(json_file)
        print("\n" + json_data["question_str"] + "\n")

if __name__ == "__main__":
    main()
