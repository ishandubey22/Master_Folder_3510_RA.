# Master_Folder_3510_RA.
1. A link to Ruiquian Gao et al.'s paper is attached here for cross-referencing the implementation - https://rqgao0422.github.io/assets/pdf/practical_algorithms_and_validated_incentives_for_A_CEEI.pdf
2. RA-CEEI.py contains a simplified implementation of the algorithm. a. Specifically, the implementation does not account for EF-TB. Saves on computational resources. b. All other features of the RA-CEEI algorithm are implemented as is. c. We have constructed a test case already, which can be modified to observe the algorithm's behaviour under different conditions. d. Limited iterations to 500 to not let the algorithm keep on running indefinitely (as ILP is NP-hard). We can increase the limit based on computational powers of our system.
3. (Research Form (Responses).xlsx) - The Excel Sheet with raw data for my survey's responses, along with a few calculations/statistics also present.
4. Require Python 3.7 or higher due to the dataclass module.
