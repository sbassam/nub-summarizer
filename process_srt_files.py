from os import listdir, walk
from os.path import isfile, join
import re
import argparse

class SrtProcessor(object):
    def __init__(self, lesson_dir, lesson_name, output_dir):
        self.lesson_dir = lesson_dir
        self.lesson_name = lesson_name
        self.output_dir = output_dir
        self.video_titles = []

    def get_lessons(self):
        l = [f for f in listdir(self.lesson_dir) if isfile(join(self.lesson_dir, f))]
        self.video_titles = sorted(l)

    def process(self):
        self.get_lessons()
        text = ''
        for i in range(len(self.video_titles)):
            path = join(self.lesson_dir, self.video_titles[i])
            file = open(path, "r")
            lines = file.readlines()
            file.close()

            for line in lines: # regex from stackoverflow #51073045
                if re.search('^[0-9]+$', line) is None and re.search('^[0-9]{2}:[0-9]{2}:[0-9]{2}',
                                                                     line) is None and re.search('^$', line) is None:
                    text += ' ' + line.rstrip('\n')
                text = text.lstrip()
        out_path = join(self.output_dir, self.lesson_name+'.txt')
        text_file = open(out_path, "w")
        text_file.truncate()
        text_file.write(text)
        text_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert SRT files to single line Text file for fine-tuning')
    parser.add_argument('--output_dir', action="store", default='./data/processed_lessons/')
    parser.add_argument('--lessons_dir', action="store", default='data/raw_lessons')
    args = parser.parse_args()
    lessons_dir, output_dir = args.lessons_dir, args.output_dir
    lessons_list = next(walk(lessons_dir))[1]
    for i in lessons_list:
        lesson_path = join(lessons_dir, i)
        s = SrtProcessor(lesson_path, i, output_dir)
        s.process()

