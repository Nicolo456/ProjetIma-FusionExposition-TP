import rawpy
import imageio
import os


def convert_arw_to_tiff(input_folder, output_folder):
    """
    Converts all .ARW files in the input folder to .tiff format and saves them in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing .ARW files.
    - output_folder (str): Path to the folder where .tiff files will be saved.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each .ARW file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.arw'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(file_name)[0]}.tiff")

            # Open and process the RAW file
            with rawpy.imread(input_path) as raw:
                rgb_image = raw.postprocess()

            # Save the processed image as TIFF
            imageio.imwrite(output_path, rgb_image)
            print(f"Converted: {file_name} -> {output_path}")


if __name__ == "__main__":
    input_folder = "img/foyer_raw"
    output_folder = "img/foyer"
    convert_arw_to_tiff(input_folder, output_folder)
