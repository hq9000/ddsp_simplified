import os, argparse, yaml

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


from dataloader import make_supervised_dataset
from train_utils import make_supervised_model, create_callbacks


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Supervised Training Parameters.')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to config file.')   
    args = parser.parse_args()

    # Read the config
    with open(args.path) as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))
    
    # Create the datasets and the preprocessor
    train_set, validation_set, _ = make_supervised_dataset(config['data']['path'],
                                                    mfcc=config['model']['encoder'],
                                                    batch_size=config['training']['batch_size'],
                                                    sample_rate=config['data']['sample_rate'])
    print('Dataset created.')
    
    # Create the model and define the training 
    model = make_supervised_model(config)
    optimizer = Adam(learning_rate=ExponentialDecay(config['optimizer']['lr'],
                                decay_steps=config['optimizer']['decay_steps'],
                                decay_rate=config['optimizer']['decay_rate']))

    # Model Saving and Experiment Tracking
    callbacks = create_callbacks(config, monitor='val_spec_loss')
    config['model']['path'] = callbacks[0].save_path

    # Save the config for future reference
    config_save_fpath = os.path.join(callbacks[0].save_dir, config['run_name']+'.yaml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)

    # Compile and train
    model.compile(optimizer)
    print('Model Compiled.')
    history = model.fit(train_set,
                        validation_data=validation_set,
                        callbacks=callbacks,
                        epochs=config['training']['epochs'])                        