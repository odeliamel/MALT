import torch


def replicate_input(x):
    return x.detach().clone()


def to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


def log_success_indices(clean_x_test, clean_y_test, device, logger, model, x_adv):
    a_list = []
    batch_size = 1
    with torch.no_grad():
        for counter in range(clean_x_test.shape[0]):
            x_adv_PET_curr = x_adv[counter * batch_size:(counter + 1) *
                                                            batch_size].to(device)
            y_curr = clean_y_test[counter * batch_size:(counter + 1) *
                                                       batch_size].to(device)

            output_pet = model(x_adv_PET_curr)
            pet_suc = output_pet.max(1)[1] != y_curr
            # print(f"original label {y_curr}, adversarial new label {output_pet.max(1)[1]}")

            if pet_suc:
                a_list.append(counter)

            # print(counter)
            # print(f"attack succeeded for indices {a_list}")
    logger.info(counter)
    logger.info(f"attack succeeded for indices {a_list}")
    asr = len(a_list) / clean_x_test.shape[0]
    logger.info(f"ASR: {asr}, RA: {1 - asr}")
