
# All arguments are set to default values in each script.


echo "================= Data Preprocess ==================="

echo "==================Processing Diagram Data=================="
# Get the data parsed results and save to one annotation file - pgps9k_test.json
python multimodal_formalizer/diagram_parser.py --geometry3k_root datasets/geometry3k/ --pgps9k_root datasets/PGPS9K/


echo "==================Processing Text Data=================="
# Parse the problem text and update the annotation file - geometry3k_test.json
python multimodal_formalizer/text_parser.py --data_file datasets/geometry3k/geometry3k_test.json

# Parse the problem text and update the annotation file - pgps9k_test.json
python multimodal_formalizer/text_parser.py --data_file datasets/PGPS9K/pgps9k_test.json


echo "==================Processing Image Data=================="
# Annotate the problem image and gather them into one folder
python ./multimodal_formalizer/annotator.py --geometry3k_root datasets/geometry3k/ --pgps9k_root datasets/PGPS9K/

# Remove useless folders to save space

rm -rf ./datasets/PGPS9K/Diagram \
    ./datasets/PGPS9K/Diagram_Visual \
    ./datasets/PGPS9K/PGPS9K\
    ./datasets/PGPS9K/Geometry3K\
    ./datasets/geometry3k/logic_forms\
    ./datasets/geometry3k/symbols\
    ./datasets/geometry3k/test\
    ./datasets/geometry3k/train\
    ./datasets/geometry3k/val\

echo "==================Preprocessing Done=================="