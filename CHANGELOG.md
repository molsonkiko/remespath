# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - yyyy-mm-dd
 
### To Be Added

- better documentation
- code coverage site
- assignment/in-place mutation operations
- recursive search
- Publish version to PYPI
 
### To Be Changed

 
### To Be Fixed

- Improve performance, probably by reducing memory copy during boolean indexing

## [0.3.0] - 2022-05-05

Until further notice, performance is not appreciably faster than 0.2.0.

### Added

- `*` operator for getting all dict keys (also works for getting all array indices, but you could do that with `[:]` before anyway)

## [0.2.0] - 2022-05-05

Appears to be bug-free for most if not all reasonable queries. Passes the full test suite.

Performance is still very slow compared to e.g. gorp.jsonpath (see my other package, gorpy) for the subset of queries that can be executed in those other implementations.

### Fixed 

- Many bugs

### Added

- Projections (e.g., `search('@{@.foo, @.bar}', {'foo': 1, 'bar': 2})` returns `[1, 2]`)

## [0.1.0] - 2022-04-16

### Added

- Everything!