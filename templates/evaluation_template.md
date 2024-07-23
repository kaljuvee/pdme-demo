<prefix><user_start>Here is a prompt:
{
    "instruction": \"""<question_full>\""",
}

Here are the outputs of the models:
[
    {
        "model": 1,
        "answer": \"""<response1>\"""
    },
    {
        "model": 2,
        "answer": \"""<response2>\"""
    }
]

Please create a dict containting the highest quality answer, i.e., produce the following output:

{
  'best_model': <model-name>
}

Please provide the response that the majority of humans would consider better.

<assistant_start>{
  'best_model': 