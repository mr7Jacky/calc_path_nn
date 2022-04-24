import csv
import torch
import pandas as pd


def calculate_class_correct(true_label, pred_label):
    """
    Returns: 1 if model correctly predicted image, 0 if incorrect
    """
    if true_label == pred_label:
        return 1
    return 0


def test_model(images, labels, i, model):
    """
    Put image through model to get predicted class

    Args:
        images: Images from the dataloader
        labels: Labels from the dataloader
        i: The i'th image wanting to be tested
        model: The NN model

    Returns:  The  model predicted class, the True Image class,
    """
    img = images[i]
    img = img.unsqueeze(1)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))  # What the machine guesses
    true_label = labels.numpy()[i].item()  # What the image is
    # Get the probability associated with the true label
    probability_list = probab

    return pred_label, true_label, probability_list


def class_tracking(model, dataloader, filename):
    """
    Record image csv outcomes for a single model to a csv file

    Args:
        model: The model wanting to be tested
        dataloader: The data loader containing the images
        filename: Name of the output file

    """
    print("Outputting to csv file...")
    file_name = f"{filename}"

    with open(f"test_results/{file_name}.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Image #", "True Class", "Predicted Class", "Correct"])

        for images, labels in dataloader:  # Loops for each batch
            for i in range(len(labels)):  # Loops through each batch

                true_label, pred_label = test_model(images, labels, i, model)

                output = calculate_class_correct(true_label, pred_label)

                # Write image statistics to csv file
                output = [str(i), str(true_label), str(pred_label), str(output), ""]
                writer.writerow(output)
    print(f"Finished Outputting to {filename} file")
    file.close()


def csv_class_statistics(experiment, data):
    """
    Record the model's ability to predict the different classes into a csv file for all models

    """
    exp = experiment
    state = exp.state
    if data.lower() == "val":
        temp = "Val"
        testdata = experiment.val_dl
    else:
        temp = "Train"
        testdata = experiment.train_dl
    class_df = pd.read_csv(rf'MNIST-{exp.compression}-ClassBreakdown.csv')
    # Format for writing to final csv file and create lists for statistics recording

    temp_class_df = pd.DataFrame(columns=[f"For {temp}", "Correctly Predicted Images", "0's", "1's", "2's", "3's", "4's", "5's", "6's", "7's", "8's", "9's"])
    temp_class_df["Correctly Predicted Images"] = [f"For {experiment.state}:"]
    orig_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for images, labels in testdata:  # Loops for each batch
        for i in range(len(labels)):  # Loops through each batch
            pred, true, true_prob = test_model(images, labels, i, experiment.model)
            if calculate_class_correct(true, pred):
                orig_c[true] += 1

    for i in range(10):
        temp_class_df[f"{i}'s"] = [orig_c[i]]

    temp_class_df.append(pd.Series(), ignore_index=True)
    class_df.append(temp_class_df)
    og = pd.concat([class_df, temp_class_df])
    og.to_csv(f'MNIST-{exp.compression}-ClassBreakdown.csv', index=False)


def csv_load(experiment, data):
    exp = experiment
    model = experiment.model
    if data.lower() == "val":
        temp = "Val"
        testdata = experiment.val_dl
    else:
        temp = "Train"
        testdata = experiment.train_dl
    state = experiment.state
    # print(f'{exp.model_name}-{temp}-{exp.compression}-ClassBreakdown.csv')

    # Structure class Specific dataframe
    class_df = pd.read_csv(rf'MNIST-{exp.compression}-ClassBreakdown.csv')
        # Format for writing to final csv file and create lists for statistics recording

    temp_class_df = pd.DataFrame(columns=[f"{temp} Correct:", "Correctly Predicted Images", "0's", "1's", "2's", "3's", "4's", "5's", "6's", "7's", "8's", "9's"])
    temp_class_df["Correctly Predicted Images"] = [f"For {experiment.state}:"]
    orig_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    percent_c = ["", "Percent Correct:"]
    #

    # structure giant big spreadsheet dataframe
    df_output = []
    df_label = []
    df_true_prob = []
    df_predicted_prob = []
    #

    # Big loop to run each image through the model and collect statistics
    for images, labels in testdata:  # Loops for each batch
        for i in range(len(labels)):  # Loops through each batch
            pred, true, probability_list = test_model(images, labels, i, model)
            df_label.append(true)
            df_output.append(pred)
            df_true_prob.append(round(probability_list[true], 4) * 100)
            df_predicted_prob.append(round(probability_list[pred], 4) * 100)
            if calculate_class_correct(true, pred):
                orig_c[true] += 1
            total_c[true] += 1

    # print(len(df_output))

    for i in range(len(orig_c)):
        percent_c.append("%.2f %%" % (100 * (orig_c[i] / total_c[i])))

    # Output to class specific dataframe csv file
    for i in range(10):
        temp_class_df[f"{i}'s"] = [orig_c[i]]

    to_append = percent_c
    df_length = len(temp_class_df)
    temp_class_df.loc[df_length] = to_append

    class_df.append(temp_class_df)
    og = pd.concat([class_df, temp_class_df])
    og.to_csv(f'MNIST-{exp.compression}-ClassBreakdown.csv', index=False)
    #

    # Output to giant dataframe csv file
    df = pd.read_csv(rf'MNIST-{exp.compression}-Overview.csv')
    # print(len(df))

    df[f"{state}"] = df_output
    df[f"True Label Probability - {state}"] = df_true_prob
    df[f"Pred Label Probability - {state}"] = df_predicted_prob
    df["Image Label"] = df_label
    df.to_csv(f'MNIST-{exp.compression}-Overview.csv', index=False)
    #


def csv_initialize(exp, data):
    # Create general csv file
    df = pd.DataFrame(columns=["Train/Val", "Image Index", "Image Label"])
    if data.lower() == 'val':
        temp = "Val"
        df["Image Index"] = [i for i in range(10000)]
    else:
        temp = "Train"
        df["Image Index"] = [i for i in range(60000)]
    df["Train/Val"] = f"{temp}"
    df.to_csv(f'MNIST-{exp.compression}-Overview.csv', index=False)
    # Create model specific class performance csv file
    df = pd.DataFrame(columns=[f"For {temp}", "Correctly Predicted Images", "0's", "1's", "2's", "3's", "4's", "5's", "6's", "7's", "8's", "9's"])
    df.to_csv(f'MNIST-{exp.compression}-ClassBreakdown.csv', index=False)


def csv_mnist_finish(experiment):
    exp = experiment
    print("Finishing up csv file...")
    testdata = experiment.val_dl
    df = pd.read_csv(rf'MNIST-{exp.compression}-Overview.csv')

    full_list = []
    for images, labels in testdata:  # Loops for each batch
        for i in range(len(labels)):  # Loops through each batch
            i_list = []
            for j in range(len(images[i].numpy()[0])):  # Loops through each image
                for k in range(len(images[0].numpy()[0][j])):  # Loops through each pixel row
                    i_list.append(images[i].numpy()[0][j][k])
            full_list.append(i_list)

    temp = pd.DataFrame(full_list)
    result = pd.concat([df, temp], axis=1)
    result.to_csv(f'MNIST-{exp.compression}-Overview.csv', index=False)

