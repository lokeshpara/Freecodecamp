import matplotlib.pyplot as plt
import numpy as np

from utils import denormalize

def plot_images(img_data, classes, img_name):
  figure = plt.figure(figsize=(10, 10))
  
  num_of_images = len(img_data)
  for index in range(1, num_of_images + 1):
      img = denormalize(img_data[index-1]["img"])  # unnormalize
      plt.subplot(5, 5, index)
      plt.axis('off')
      plt.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
      plt.title("Predicted: %s\nActual: %s" % (classes[img_data[index-1]["pred"]], classes[img_data[index-1]["target"]]))
  
  plt.tight_layout()
  plt.savefig(img_name)

def plot_graph(data, metric):
    fig, ax = plt.subplots()

    for sub_metric in data.keys():
      ax.plot(data[sub_metric], label=sub_metric)
    
    plt.title(f'Change in %s' % (metric))
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    
    ax.legend()
    plt.show()

    fig.savefig(f'%s_change.png' % (metric.lower()))