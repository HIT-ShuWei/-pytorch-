class Solution:
    def combinationSum(self, candidates, target: int):
        # 回溯 + 剪枝


        def backstracking(candidates, target, cur_sum, start_index):
            '''
            参数设定
            candidates - 可选数字
            target - 目标数字
            cur_sum - 当前数组内已经和
            start_index - 在candidates里开始搜索的位置index
            #关于start_index：在组合问题中，如果是在一个集合中找组合，就需要start，多个集合中找组合就不需要
            '''
            if cur_sum == target:
                # 终止条件：如果当前为目标和，直接存储
                self.res.append(self.path.copy())
                return
            if cur_sum > target:
                # 剪枝：如果当前和已经大于目标，直接返回
                return

            for i in range(start_index, len(candidates)):
                self.path.append(candidates[i])
                cur_sum += candidates[i]
                if cur_sum > target:
                    # 剪枝操作，对于排序好的candidate，如果当前元素已经超处范围，那么后序一定超，全部剪掉即可
                    break
                backstracking(candidates, target, cur_sum, i)  # 递归
                cur_sum -= candidates[i]  # 回溯
                self.path.pop()  # 回溯
            return


        candidates.sort()  # 将candidates进行排序，为了后序的剪枝操作
        self.res = []  # 用于存储最终的结果
        self.path = []  # 用于存储单个叶节点
        backstracking(candidates, target, 0, 0)
        return self.res

if __name__ == '__main__':
    solution = Solution()
    candidate = [2,3,6,7]
    target = 7
    ans = solution.combinationSum(candidate,target)
    print(ans)