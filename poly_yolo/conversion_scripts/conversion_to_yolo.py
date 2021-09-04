import json

if __name__ == '__main__':
    class_mapping = {'bike': '0',
                     'bus': '1',
                     'car': '2',
                     'motor': '3',
                     'person': '4',
                     'rider': '5',
                     'traffic light': '6',
                     'traffic sign': '7',
                     'train': '8',
                     'truck': '9'
                     }

    out = open('annotation.txt', 'w')

    with open('a.json') as json_file:
        data = json.load(json_file)

        for p in data:
            outputlineFile = p['name'] + ' '
            outputlineData = ''

            for l in p['labels']:
                if 'box2d' in l:
                    outputlineData = outputlineData + str(round(l['box2d']['x1'])) + ',' + \
                                                  str(round(l['box2d']['y1'])) + ',' + \
                                                  str(round(l['box2d']['x2'])) + ',' + \
                                                  str(round(l['box2d']['y2'])) + ','

                    if l['category'] in class_mapping:
                        outputlineData = outputlineData + class_mapping[l['category']] + ','
                    else:
                        print('Error! Undefined class ', l['category'], ' in picture ', p['name'])

                        outputlineData = ''
                        continue

                    # No polyline available in the current data set. Just copy box2d information.
                    outputlineData = outputlineData + str(round(l['box2d']['x1'])) + ',' + \
                                                      str(round(l['box2d']['y1'])) + ',' + \
                                                      str(round(l['box2d']['x2'])) + ',' + \
                                                      str(round(l['box2d']['y2'])) + ' '

            print(outputlineFile + outputlineData[:-1], file=out)

            outputline = ''

    out.close()