# Malicious websites blocking system using Deep Learning algorithms

The project consists of a malicious website blocking system. The blocking system is created in python and uses Deep Learning to train the tool.

## Structure

In the **config** folder, you will find the different configurations for each network to execute.

In the ```job.sh```script the main project executen such as its configuration is depicted.

In the **wbs** folder the main project is depicted. 

In the **results** folder, all the results of the executions can be found.


## Execution commands 

To execute the whole project:

**Note:** look at the main.py script to decide which execution to follow and its parameters (--config)

```bash
python -m wbs --config config.json
```

The project can be also executed from a shell script named job.sh:

```bash
chmod +x job_name.sh

./job_name.sh
```
