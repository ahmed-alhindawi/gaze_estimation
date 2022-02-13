def print_args(args, key_width=30, value_width=50):
    print("|".join([str("{:^" + str(key_width) + "}").format("Key"), str("{:^" + str(value_width) + "}").format("Value")]))
    print("=" * (key_width + value_width))
    for k, v in sorted(vars(args).items()):
        print(":".join([str("{:<" + str(key_width) + "}").format(k[:key_width - 3] + "..." if len(k) >= key_width else k),
                        str("{:<" + str(value_width) + "}").format(str(v)[:value_width - 3] + "..." if len(str(v)) >= value_width else v)]))
    print("-" * (key_width + value_width))