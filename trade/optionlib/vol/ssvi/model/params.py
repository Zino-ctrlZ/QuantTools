

from pydantic import Field
from pydantic.dataclasses import dataclass
from trade.helpers.Logging import setup_logger
logger = setup_logger('optionlib.ssvi.model.params')

@dataclass(slots=True)
class SSVIModelParams:
    """
    SSVI Model Parameters for the Stochastic Volatility Surface.
    This class holds the parameters for the SSVI model, including the ATM variance, 
    long-term variance, speed of mean reversion, skewness, kurtosis, and correlation.
    Attributes:
        var0_hat (float): Initial variance estimate at ATM.
        var_inf_hat (float): Long-term variance estimate.
        kappa_hat (float): Speed of mean reversion.
        eta_hat (float): Skewness parameter.
        lambda_hat (float): Kurtosis parameter.
        rho_hat (float): Correlation parameter.
        atm_loss (float): Loss associated with the ATM volatility estimation.
        surface_loss (float): Loss associated with the surface fitting.
    """
    var0_hat: float = Field(default=0.0, description="Initial variance estimate at ATM")
    var_inf_hat: float = Field(default=0.0, description="Long-term variance estimate")
    kappa_hat: float = Field(default=0.0, description="Speed of Mean Reversion")
    eta_hat: float = Field(default=0.0, description="Skewness parameter")
    lambda_hat: float = Field(default=0.0, description="Kurtosis parameter")
    rho_hat: float = Field(default=0.0, description="Correlation parameter")
    atm_loss: float = Field(default=0.0, description="Loss associated with ATM volatility estimation")
    surface_loss: float = Field(default=0.0, description="Loss associated with surface fitting")
    nrmse: float = Field(default=0.0, description="Normalized Mean Squared Error")
    rw_nrmse: float = Field(default=0.0, description="Right Wing Normalized Mean Squared Error")
    lw_nrmse: float = Field(default=0.0, description="Left Wing Normalized Mean Squared Error")
    nmae: float = Field(default=0.0, description="Normalized Mean Absolute Error")
    rw_nmae: float = Field(default=0.0, description="Right Wing Normalized Mean Absolute Error")
    lw_nmae: float = Field(default=0.0, description="Left Wing Normalized Mean Absolute Error")
    
    def __repr__(self):
        acceptable_fields = ['var0_hat', 'var_inf_hat', 'kappa_hat',
                                'eta_hat', 'lambda_hat', 'rho_hat',
                                'atm_loss', 'surface_loss']
        params = {field: getattr(self, field) for field in acceptable_fields}
        return (f"SSVIModelParams{params}\n")