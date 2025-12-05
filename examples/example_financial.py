"""Example demonstrating financial trading environment with backtesting.

NOTE: This example requires Alpaca API credentials to download financial data.
You must set the following environment variables:
    - ALPACA_API_KEY: Your Alpaca API key
    - ALPACA_SECRET_KEY: Your Alpaca secret key

You can set these in a .env file in the project root or export them in your shell.
"""

if __name__ == "__main__":
    import numpy as np

    import stable_worldmodel as swm

    ######################
    ##  World Creation  ##
    ######################

    print("Creating Financial Backtesting World...")
    world = swm.World(
        "swm/Financial-v0",
        num_envs=2,
        render_mode=None,
    )

    print("\n" + "=" * 60)
    print("Environment Information:")
    print("=" * 60)
    print(f"Available variations: {world.single_variation_space.names()}")
    print(f"Action Space: {world.envs.action_space}")
    print("Actions: 0=SELL, 1=BUY, 2=HOLD")

    #######################
    ##  Random Trading   ##
    #######################

    print("\n" + "=" * 60)
    print("Running Random Trading Episodes...")
    print("=" * 60)

    # Set a random policy
    world.set_policy(swm.policy.RandomPolicy())

    # Run a few episodes manually
    world.reset(seed=42)

    for episode in range(2):
        print(f"\n--- Episode {episode + 1} ---")
        world.reset(seed=42 + episode)

        episode_rewards = []
        for step in range(50):  # Run 50 steps per episode
            actions = world.policy.get_action(world.states)
            world.states, rewards, terminated, truncated, world.infos = world.envs.step(actions)
            episode_rewards.append(rewards)

            if step % 20 == 0:
                print(
                    f"Step {step}: Avg Portfolio Value = ${np.mean([world.infos['portfolio_value'][i] for i in range(2)]):,.2f}"
                )

            if all(terminated) or all(truncated):
                break

        print(f"Episode finished: Total Reward = {np.sum(episode_rewards):.4f}")

    #############################
    ##  Backtest Analysis      ##
    #############################

    print("\n" + "=" * 60)
    print("Detailed Backtest Metrics (Single Environment):")
    print("=" * 60)

    # Access the first environment from the world's vectorized environments
    env = world.envs.envs[0]
    env.reset(seed=42)

    # Run a short episode
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Get comprehensive backtest results
    backtest_results = env.unwrapped.get_backtest_results()

    print("\nReturns Metrics:")
    if "total_return" in backtest_results:
        print(f"  Total Return: {backtest_results['total_return']:.2%}")
    if "annualized_return" in backtest_results:
        ann_ret = backtest_results["annualized_return"]
        if np.isfinite(ann_ret):
            print(f"  Annualized Return: {ann_ret:.2%}")
        else:
            print(f"  Annualized Return: {ann_ret}")

    print("\nRisk Metrics:")
    for metric in ["sharpe_ratio", "sortino_ratio", "max_drawdown", "volatility"]:
        if metric in backtest_results:
            val = backtest_results[metric]
            if "ratio" in metric:
                print(f"  {metric.replace('_', ' ').title()}: {val:.3f}")
            else:
                print(f"  {metric.replace('_', ' ').title()}: {val:.2%}")

    print("\nTrading Statistics:")
    if "total_trades" in backtest_results:
        print(f"  Total Trades: {backtest_results['total_trades']}")
    if "win_rate" in backtest_results:
        print(f"  Win Rate: {backtest_results['win_rate']:.2%}")

    print(f"\nSample available metrics: {list(backtest_results.keys())[:8]}")

    print("\n" + "=" * 60)
    print("Financial trading example completed successfully!")
    print("=" * 60)
