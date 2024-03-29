from utils.prepare_data import process_all_images

if __name__ == "__main__":
    input_dir = './raw_data/'
    output_dir = './prepared_data/'
    process_all_images(input_dir, output_dir)