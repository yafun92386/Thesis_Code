import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def rdLoss(save_path):
	with open(save_path + "/total_loss.txt") as f:
		content = f.read().splitlines()
		title = content[0]
		content = content[1:]
		content = [float(loss) for loss in content]
	return title, content

def showPlot(title, loss_data, save_path):
	fig, ax = plt.subplots()
	# this locator puts ticks at regular intervals
	xloc = ticker.MultipleLocator(base=5.0)
	yloc = ticker.MultipleLocator(base=0.1)
	ax.xaxis.set_major_locator(xloc)
	ax.yaxis.set_major_locator(yloc)
	
	# Add titles
	plt.title(title)
	plt.xlabel("Epoch")
	plt.ylabel("NLLLoss")

	count = 0
	# Add notation
	for x, y in zip(range(1, len(loss_data)+1), loss_data):
		if count%2 == 0:
			if count%5 == 0:
				plt.text(x+0.02, y+0.02, '%.3f'%y, ha='center', va='bottom',fontsize=6, color='red')
			else:
				plt.text(x+0.02, y+0.02, '%.3f'%y, ha='center', va='bottom',fontsize=6)
		else:
			if count%5 == 0:
				plt.text(x+0.02, y+0.1, '%.3f'%y, ha='center', va='bottom',fontsize=6, color='red')
			else:
				plt.text(x+0.02, y+0.1, '%.3f'%y, ha='center', va='bottom',fontsize=6)
		count+=1

	plt.plot(range(1, len(loss_data)+1), loss_data, marker='o')
	plt.plot(range(5, len(loss_data)+1, 5), loss_data[4:len(loss_data)+1:5], marker='o', color='red')
	plt.savefig(save_path + "/total_loss.jpg")
	plt.show()


if __name__ == "__main__":

	save_dir = '4_emb_seg_VAE'
	save_path = os.path.join(save_dir)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# print(save_path)
	title, content = rdLoss(save_path)
	# print(content)
	showPlot(title, content, save_path)

	pass