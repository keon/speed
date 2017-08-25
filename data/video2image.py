import cv2

def video2image(video_file, train=True, split_ratio=0.9):
    video_inp = video_file + '.mp4'
    video_reader = cv2.VideoCapture(video_inp)
    image_folder = video_file + '/images/'

    if train:
        # valid_image = 'valid/images/'
        label_inp = open(video_file + '.txt').readlines()
        train_label = open(video_file + '/labels.txt', 'w')
        # valid_label = open('valid/labels.txt', 'w')
    print('imagefying video: ', video_file)
    counter = 0
    while(True):
        if counter % 100 == 0:
            print('processing...', counter)
        ret, frame = video_reader.read()
        if ret == True:
            if False and train and 0 <= counter < 2040:
                cv2.imwrite(valid_image + str(counter).zfill(6) + '.png', frame)
                if train:
                    valid_label.write(label_inp[counter])
            else:
                cv2.imwrite(image_folder + str(counter).zfill(6) + '.png', frame)
                train_label.write(label_inp[counter])
            counter += 1
        else:
            break

    video_reader.release()
    if train:
        train_label.close()

if __name__ == '__main__':
    # create training / validation data
    video2image('train', train=True, split_ratio=0)
    # create testing data
    # video2image('test', train=False)
