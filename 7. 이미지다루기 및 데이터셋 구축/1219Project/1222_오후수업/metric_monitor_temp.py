from collections import defaultdict

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.rest()

    def rest(self):
        self.metrics = defaultdict(lambda: {'val': 0, "count": 0, "avg": 0})

    def update(self, metrics_name, val):
        metrics = self.metrics[metrics_name]

        metrics["val"] += val
        metrics["count"] += 1
        metrics["avg"] = metrics["val"] / metrics["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name} : {avg.{float_precision}f}".format(
                    metric_name=metric_name, avg=metrics["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metrics) in self.metrics.items()
            ]
        )