{
  inputs = {
    nixpkgs.url = "github:dpaetzel/nixpkgs/dpaetzel/nixos-config";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs =
    {
      self,
      nixpkgs,
      devenv,
      systems,
      ...
    }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python311;
          mlflowFixed = python.pkgs.mlflow.overridePythonAttrs (oldAttrs: {
            # mlflow requires setuptools at runtime but does not specify so in its
            # package definition.
            propagatedBuildInputs = oldAttrs.propagatedBuildInputs ++ [
              python.pkgs.setuptools
              python.pkgs.psycopg2
            ];
          });
        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                packages = [ mlflowFixed ];
                languages.python.enable = true;
                languages.python.package = python;

                # https://devenv.sh/reference/options/
                languages.julia.enable = true;
              }
            ];
          };
        }
      );
    };
}
